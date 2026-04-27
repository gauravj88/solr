/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.solr.search.function;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.ReaderUtil;
import org.apache.lucene.queries.function.FunctionValues;
import org.apache.lucene.queries.function.ValueSource;
import org.apache.lucene.queries.function.docvalues.FloatDocValues;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.solr.handler.component.ResponseBuilder;
import org.apache.solr.request.SolrQueryRequest;
import org.apache.solr.request.SolrRequestInfo;
import org.apache.solr.search.DocSet;
import org.apache.solr.search.ExtendedQuery;
import org.apache.solr.search.PostFilter;
import org.apache.solr.search.SolrIndexSearcher;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Linearly scales {@code source} into {@code [targetMin, targetMax]} using bounds derived from the
 * <b>current request's matching DocSet</b>.
 *
 * <p>Differs from Lucene's {@code ScaleFloatFunction} in three ways:
 *
 * <ul>
 *   <li>{@code maxObs} is computed over only the request's matching set (intersection of
 *       {@code q} and all non-PostFilter {@code fq}s), not every doc in every segment. For
 *       narrowly filtered queries this is orders of magnitude faster.
 *   <li>{@code minObs} is anchored at {@code 0} when the observed minimum is positive. This
 *       prevents the lowest-matching doc from squashing to {@code targetMin} for non-negative
 *       sources (Lucene scores, recency boosts, etc.). With the anchor, only {@code raw=0}
 *       (typically a non-matching doc seen via {@code expand}) maps to {@code targetMin}; every
 *       matching doc spreads across {@code (targetMin, targetMax]}. For sources that legitimately
 *       produce negative values, the observed minimum is preserved.
 *   <li>Output is clamped to {@code [targetMin, targetMax]}.
 * </ul>
 *
 * <p>Falls back to a full index scan when a Solr request context is not available — e.g. when
 * invoked from Lucene-level tests or embedded tool usage.
 *
 * <h3>Known edge cases handled</h3>
 * <ul>
 *   <li><b>PostFilter exclusion:</b> PostFilters (e.g. {@code {!collapse}}) are stripped from
 *       the filter list before computing the DocSet to prevent reentrant evaluation.</li>
 *   <li><b>ExtendedQuery with post-filter cost:</b> Filters implementing {@link ExtendedQuery}
 *       with {@code getCost() >= 100} are also stripped, as Solr treats them as post-filters.</li>
 *   <li><b>Reentrant guard:</b> Depth-counted guard prevents infinite recursion when
 *       {@code matchset_scale} appears inside a query being materialized.</li>
 *   <li><b>Request-level bounds cache:</b> Ensures collapse sort, expand sort, top-level sort,
 *       and fl display all see identical bounds within the same request. Synchronized writes
 *       to the cache map (Solr's {@code req.getContext()} is a plain HashMap) make it safe to
 *       use under Solr's parallel {@code expand} / multi-segment fl evaluation.</li>
 *   <li><b>Missing docvalues:</b> Both fast and slow paths skip docs without a value via
 *       {@code FunctionValues#exists(int)}, keeping bounds consistent across paths.</li>
 *   <li><b>All-equal values (without anchor):</b> If the source can produce only one value
 *       across the matching set AND that value is non-positive, returns {@code targetMin}.</li>
 *   <li><b>Empty matching set:</b> When no docs match, returns {@code targetMin}.</li>
 *   <li><b>NaN/Inf filtering:</b> Skipped during bounds computation and clamped during value production.</li>
 *   <li><b>No request context:</b> Falls back to full index scan gracefully.</li>
 * </ul>
 */
public class MatchSetScaleFloatFunction extends ValueSource {

  private static final Logger log = LoggerFactory.getLogger(MatchSetScaleFloatFunction.class);

  protected final ValueSource source;
  protected final float targetMin;
  protected final float targetMax;

  public MatchSetScaleFloatFunction(ValueSource source, float targetMin, float targetMax) {
    this.source = source;
    this.targetMin = targetMin;
    this.targetMax = targetMax;
  }

  @Override
  public String description() {
    return "matchset_scale(" + source.description() + "," + targetMin + "," + targetMax + ")";
  }

  // =========================================================================
  //  Internal bounds holder — immutable after construction
  // =========================================================================

  private static final class Bounds {
    final float min;
    final float max;

    Bounds(float min, float max) {
      this.min = min;
      this.max = max;
    }
  }

  // =========================================================================
  //  Constants for request-level caching and reentrant guard
  // =========================================================================

  /**
   * Key used in the per-request context to guard against reentrant DocSet
   * materialization.  Value is an Integer depth counter, not a boolean,
   * so that nested matchset_scale calls don't prematurely clear the guard.
   */
  private static final String COMPUTE_GUARD_KEY = "matchset_scale.computing";

  /**
   * Prefix for per-request bounds cache keys.
   */
  private static final String BOUNDS_CACHE_PREFIX = "matchset_scale.bounds:";

  // =========================================================================
  //  Helper methods
  // =========================================================================

  @Override
  public void createWeight(Map<Object, Object> context, IndexSearcher searcher) throws IOException {
    source.createWeight(context, searcher);
  }

  /**
   * Build a cache key unique to this (source, targetMin, targetMax) triple
   * so that different matchset_scale invocations in the same request don't
   * share bounds incorrectly.
   */
  private String boundsCacheKey() {
    return BOUNDS_CACHE_PREFIX + source.hashCode() + ":" + targetMin + ":" + targetMax;
  }

  /**
   * Returns the per-request context map, or {@code null} if no Solr
   * request is active.
   */
  private static Map<Object, Object> getRequestContext() {
    SolrRequestInfo reqInfo = SolrRequestInfo.getRequestInfo();
    if (reqInfo == null) return null;
    SolrQueryRequest req = reqInfo.getReq();
    if (req == null) return null;
    return req.getContext();
  }

  /**
   * Checks whether a filter query is a PostFilter or an ExtendedQuery
   * configured with post-filter cost (>= 100).  Such queries cannot
   * produce a cacheable DocSet and must be excluded from bounds computation.
   *
   * <p>Why this matters: PostFilters like {@code {!collapse}} work by
   * wrapping the collector during document collection.  If included in
   * {@code sis.getDocSet(filters)}, the collapse filter tries to evaluate
   * its sort function — which may contain this very {@code matchset_scale}
   * call — causing reentrant DocSet materialization.  The reentrant guard
   * catches this and falls back to a full-index scan, producing wrong
   * bounds that destroy sort discrimination.
   *
   * @param fq the filter query to test
   * @return true if the query should be skipped during DocSet computation
   */
  private static boolean isPostFilter(Query fq) {
    if (fq == null) return true;

    // Case 1: Direct PostFilter implementation (e.g. CollapsingQParserPlugin)
    if (fq instanceof PostFilter) {
      return true;
    }

    // Case 2: ExtendedQuery with cost >= 100 — Solr treats these as
    // post-filters even if they don't directly implement PostFilter
    if (fq instanceof ExtendedQuery) {
      ExtendedQuery eq = (ExtendedQuery) fq;
      if (eq.getCost() >= 100) {
        return true;
      }
    }

    return false;
  }

  /**
   * Returns true if the float value is NaN or Infinity.
   * Uses raw bit inspection: exponent bits all-1 means NaN or ±Inf.
   */
  private static boolean isNaNOrInf(float v) {
    return (Float.floatToRawIntBits(v) & (0xff << 23)) == (0xff << 23);
  }

  // =========================================================================
  //  Bounds computation
  // =========================================================================

  /**
   * Compute observed min/max of the source over the matching DocSet (or
   * the full index as a fallback).  Results are cached both in the
   * per-ValueSource context and in the per-request context.
   */
  private Bounds computeBounds(Map<Object, Object> vsContext, LeafReaderContext readerContext)
      throws IOException {

    List<LeafReaderContext> leaves = ReaderUtil.getTopLevelContext(readerContext).leaves();
    DocSet matchSet = findMatchSet();

    float minVal = Float.POSITIVE_INFINITY;
    float maxVal = Float.NEGATIVE_INFINITY;

    if (matchSet != null) {
      // ---- Fast path: iterate only over the matching DocSet ----
      if (matchSet.size() == 0) {
        log.debug("matchset_scale: empty matching set, returning default bounds");
        return storeBounds(vsContext, 0f, 0f);
      }

      for (LeafReaderContext leaf : leaves) {
        DocIdSetIterator it = matchSet.iterator(leaf);
        if (it == null) continue;
        FunctionValues vals = source.getValues(vsContext, leaf);
        for (int doc = it.nextDoc(); doc != DocIdSetIterator.NO_MORE_DOCS; doc = it.nextDoc()) {
          // Skip docs without a value for this source — symmetric with the slow path
          // below.  Without this check, missing-docvalue docs contribute their default
          // (typically 0) to bounds, which is inconsistent with the fallback branch.
          if (!vals.exists(doc)) continue;
          float v = vals.floatVal(doc);
          if (isNaNOrInf(v)) continue;
          if (v < minVal) minVal = v;
          if (v > maxVal) maxVal = v;
        }
      }
    } else {
      // ---- Fallback: full index scan (no request context or reentrant) ----
      log.debug("matchset_scale: no matching set available, falling back to full index scan");
      for (LeafReaderContext leaf : leaves) {
        int maxDoc = leaf.reader().maxDoc();
        FunctionValues vals = source.getValues(vsContext, leaf);
        for (int i = 0; i < maxDoc; i++) {
          if (!vals.exists(i)) continue;
          float v = vals.floatVal(i);
          if (isNaNOrInf(v)) continue;
          if (v < minVal) minVal = v;
          if (v > maxVal) maxVal = v;
        }
      }
    }

    // No valid values found at all
    if (minVal == Float.POSITIVE_INFINITY) {
      minVal = 0f;
      maxVal = 0f;
    }

    // Anchor minVal at 0 for non-negative sources (Lucene scores from query()/fuzzy/etc.,
    // and product()-of-non-negative-fields like recency boosts).
    //
    // Without this anchor, the lowest-matching doc squashes to targetMin (e.g. 0.05),
    // which is misleading: a doc that DOES match shouldn't display as "no match".
    // With the anchor, raw=0 (non-matching) maps to targetMin, and any matching doc
    // with raw>0 spreads across (targetMin, targetMax].
    //
    // For sources that can produce negative values (rare in scoring contexts), the
    // observed min is preserved.
    if (minVal > 0f) minVal = 0f;

    return storeBounds(vsContext, minVal, maxVal);
  }

  /**
   * Stores computed bounds in both the per-VS context and the per-request
   * context, then returns the Bounds object.
   */
  private Bounds storeBounds(Map<Object, Object> vsContext, float minVal, float maxVal) {
    Bounds b = new Bounds(minVal, maxVal);

    // Cache in per-ValueSource context for this evaluation pass
    vsContext.put(MatchSetScaleFloatFunction.this, b);

    // Cache in per-request context so collapse sort, expand sort,
    // top-level sort, and fl display all use the SAME bounds.
    //
    // SolrQueryRequestBase.getContext() returns a plain HashMap which is NOT
    // thread-safe.  Solr's expand component and inter-segment fl evaluation can
    // run concurrently, so synchronize on the map to prevent lost writes /
    // ConcurrentModificationException under load.
    Map<Object, Object> reqCtx = getRequestContext();
    if (reqCtx != null) {
      synchronized (reqCtx) {
        reqCtx.put(boundsCacheKey(), b);
      }
    }

    return b;
  }

  // =========================================================================
  //  Matching set resolution
  // =========================================================================

  /**
   * Locate the matching DocSet for the current request.
   *
   * <p>PostFilter queries (e.g. {@code {!collapse}}) and ExtendedQuery
   * instances with cost >= 100 are stripped from the filter list before
   * computing the DocSet.
   *
   * <p><b>Why PostFilters must be stripped:</b> Including them causes
   * {@code sis.getDocSet(filters)} to try to evaluate the collapse filter,
   * which evaluates its sort function, which contains
   * {@code matchset_scale}, which calls {@code findMatchSet()} again —
   * an infinite loop.  The reentrant guard catches this but falls back to
   * a full-index scan, producing bounds over the entire 12M+ doc index
   * instead of the actual matching set.  This compresses all matching
   * docs' scores into a tiny band (e.g. [0.05, 0.06]), destroying sort
   * discrimination.
   *
   * @return the matching DocSet, or {@code null} to signal a full-index-scan fallback
   */
  private DocSet findMatchSet() throws IOException {
    // ---- Step 1: Check if we have a request context at all ----
    SolrRequestInfo reqInfo = SolrRequestInfo.getRequestInfo();
    if (reqInfo == null) {
      return null;
    }

    ResponseBuilder rb = reqInfo.getResponseBuilder();
    if (rb == null) {
      return null;
    }

    // ---- Step 2: Prefer already-materialized results ----
    // During fl evaluation (after collection is complete), results are
    // already available.  This is the ideal case: correct bounds, no cost.
    if (rb.getResults() != null && rb.getResults().docSet != null) {
      return rb.getResults().docSet;
    }

    // ---- Step 3: Validate request prerequisites ----
    SolrQueryRequest req = reqInfo.getReq();
    if (req == null) {
      return null;
    }

    Map<Object, Object> reqCtx = req.getContext();

    // ---- Step 4: Reentrant guard check ----
    // If we are already inside a computeBounds call for this request,
    // return null to break the cycle (triggers full-index fallback).
    // Synchronize because reqCtx is a plain HashMap and other threads may
    // be writing to it via storeBounds() concurrently.
    if (reqCtx != null) {
      Object guardVal;
      synchronized (reqCtx) {
        guardVal = reqCtx.get(COMPUTE_GUARD_KEY);
      }
      if (guardVal instanceof Integer && (Integer) guardVal > 0) {
        log.debug("matchset_scale: reentrant guard fired, falling back to full scan");
        return null;
      }
    }

    SolrIndexSearcher sis = req.getSearcher();
    if (sis == null) {
      return null;
    }

    Query q = rb.getQuery();
    if (q == null) {
      return null;
    }

    // ---- Step 5: Collect non-PostFilter filter queries ----
    List<Query> filters = rb.getFilters();
    List<Query> nonPostFilters = stripPostFilters(filters);

    // ---- Step 6: Compute DocSet with depth-counted guard ----
    // The depth counter supports nested matchset_scale(a, ...) inside
    // matchset_scale(b, ...) without the inner finally block prematurely
    // clearing the outer guard.
    int prevDepth = 0;
    if (reqCtx != null) {
      synchronized (reqCtx) {
        Object val = reqCtx.get(COMPUTE_GUARD_KEY);
        prevDepth = (val instanceof Integer) ? (Integer) val : 0;
        reqCtx.put(COMPUTE_GUARD_KEY, prevDepth + 1);
      }
    }

    try {
      if (nonPostFilters == null || nonPostFilters.isEmpty()) {
        return sis.getDocSet(q);
      }
      return sis.getDocSet(q, sis.getDocSet(nonPostFilters));
    } catch (Exception e) {
      // If DocSet computation fails for any reason, fall back gracefully
      // rather than crashing the entire search request.
      log.warn("matchset_scale: failed to compute matching DocSet, falling back to full scan", e);
      return null;
    } finally {
      if (reqCtx != null) {
        synchronized (reqCtx) {
          if (prevDepth == 0) {
            reqCtx.remove(COMPUTE_GUARD_KEY);
          } else {
            reqCtx.put(COMPUTE_GUARD_KEY, prevDepth);
          }
        }
      }
    }
  }

  /**
   * Filters out PostFilter and high-cost ExtendedQuery instances from the
   * given filter list.  Returns {@code null} if no usable filters remain.
   *
   * @param filters the raw filter list from ResponseBuilder (may be null)
   * @return a new list with only standard filters, or null if empty/all-filtered
   */
  private static List<Query> stripPostFilters(List<Query> filters) {
    if (filters == null || filters.isEmpty()) {
      return null;
    }

    List<Query> result = new ArrayList<>(filters.size());
    for (Query fq : filters) {
      if (!isPostFilter(fq)) {
        result.add(fq);
      }
    }

    return result.isEmpty() ? null : result;
  }

  // =========================================================================
  //  Value production
  // =========================================================================

  @Override
  public FunctionValues getValues(Map<Object, Object> vsContext, LeafReaderContext readerContext)
      throws IOException {

    // ---- Resolve bounds with 3-level cache lookup ----
    //
    // Level 1: Per-request cache (shared across collapse, expand, sort, fl)
    //          — ensures all evaluation contexts see the same bounds
    //
    // Level 2: Per-ValueSource-context cache (same evaluation pass)
    //          — avoids recomputation across segments within one pass
    //
    // Level 3: Compute fresh bounds (first call in the request)
    //
    Bounds b = lookupBounds(vsContext);
    if (b == null) {
      b = computeBounds(vsContext, readerContext);
    }

    final float minObs = b.min;
    final float maxObs = b.max;
    final float outMin = targetMin;
    final float outMax = targetMax;
    final float obsRange = maxObs - minObs;
    final float scale = (obsRange == 0f) ? 0f : (outMax - outMin) / obsRange;

    final FunctionValues vals = source.getValues(vsContext, readerContext);

    return new FloatDocValues(this) {
      @Override
      public boolean exists(int doc) throws IOException {
        return vals.exists(doc);
      }

      @Override
      public float floatVal(int doc) throws IOException {
        // All observed values identical — return targetMin (divide-by-zero guard)
        if (obsRange == 0f) {
          return outMin;
        }

        float raw = vals.floatVal(doc);

        // Guard against NaN/Inf input producing garbage output
        if (isNaNOrInf(raw)) {
          return outMin;
        }

        float v = (raw - minObs) * scale + outMin;

        // Clamp to [targetMin, targetMax] — handles floating-point drift
        // and values outside the observed range (possible when the doc
        // wasn't in the original matching set, e.g. during expand)
        if (v < outMin) return outMin;
        if (v > outMax) return outMax;
        return v;
      }

      @Override
      public String toString(int doc) throws IOException {
        return "matchset_scale("
            + vals.toString(doc)
            + ",toMin=" + outMin
            + ",toMax=" + outMax
            + ",fromMin=" + minObs
            + ",fromMax=" + maxObs
            + ")";
      }
    };
  }

  /**
   * Look up previously computed bounds from the request-level cache first,
   * then from the per-VS-context cache.
   *
   * @return cached Bounds, or null if none found
   */
  private Bounds lookupBounds(Map<Object, Object> vsContext) {
    // Level 1: Request-level cache.  Synchronize on the HashMap because
    // storeBounds() also synchronizes on it — pairing the read with the write
    // prevents a partially-published put from being observed.
    Map<Object, Object> reqCtx = getRequestContext();
    if (reqCtx != null) {
      Object cached;
      synchronized (reqCtx) {
        cached = reqCtx.get(boundsCacheKey());
      }
      if (cached instanceof Bounds) {
        return (Bounds) cached;
      }
    }

    // Level 2: Per-VS-context cache (single-threaded per evaluation pass)
    Object cached = vsContext.get(MatchSetScaleFloatFunction.this);
    if (cached instanceof Bounds) {
      return (Bounds) cached;
    }

    return null;
  }

  // =========================================================================
  //  equals / hashCode
  // =========================================================================

  @Override
  public int hashCode() {
    int h = Float.floatToIntBits(targetMin);
    h = h * 29;
    h += Float.floatToIntBits(targetMax);
    h = h * 29;
    h += source.hashCode();
    return h ^ MatchSetScaleFloatFunction.class.hashCode();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || MatchSetScaleFloatFunction.class != o.getClass()) return false;
    MatchSetScaleFloatFunction other = (MatchSetScaleFloatFunction) o;
    return this.targetMin == other.targetMin
        && this.targetMax == other.targetMax
        && this.source.equals(other.source);
  }
}
