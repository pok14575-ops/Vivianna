from ddgs import DDGS
import trafilatura
import time
import requests
import concurrent.futures as _cf
import numpy as np
from config import (
    TAVILY_API_KEY, EXA_API_KEY, READ_MAX_CHARS,
    CROSS_ENCODER_MODEL, CROSS_ENCODER_DEVICE,
    WEB_RERANK_ENABLED, WEB_RESULTS_FETCH, WEB_FETCH_TOP_K, WEB_FETCH_MIN_PAGES,
    WEB_FETCH_TIMEOUT, WEB_CONTENT_RERANK_CHARS, WEB_RELEVANCE_FLOOR, WEB_RANK_DEBUG,
    WEB_SOURCE_DATE_ENABLED,
)

_UA = {"User-Agent": "Vivianna/1.0 (local assistant)"}

def search_web(query, max_results=3):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


def _score_relevance(query, texts):
    """Cross-encoder relevance of each text vs the query, in the same order as `texts`.

    Reuses the shared prewarmed ettin-150m instance (salience / memory rerank) — no extra
    VRAM, ~ms for a handful of pairs. Returns a list of float scores, or None when reranking
    is unavailable (disabled / <2 texts / model not loaded / any error) so callers keep their
    prior order. This is the one place that touches the model; both stages go through it."""
    if not WEB_RERANK_ENABLED or len(texts) < 2:
        return None
    try:
        from cross_encoder_model import get_model
        model = get_model(CROSS_ENCODER_MODEL, CROSS_ENCODER_DEVICE)
        if model is None:
            return None
        scores = model.predict([[query, t] for t in texts], show_progress_bar=False)
        return [float(s) for s in scores]
    except Exception as e:  # noqa: BLE001 — any failure -> caller keeps prior order
        print(f"[WEB] rerank failed ({type(e).__name__}: {e}); prior order.", flush=True)
        return None


def score_relatedness(query, passage):
    """Single-pair cross-encoder score (query vs passage), or None if unavailable. For the
    item-1 follow-up detector: 'is the current turn about the cached web topic?'. Reuses the
    shared prewarmed ettin instance (no extra VRAM). _score_relevance needs >=2 texts; this is
    the one-pair variant."""
    if not WEB_RERANK_ENABLED:
        return None
    try:
        from cross_encoder_model import get_model
        model = get_model(CROSS_ENCODER_MODEL, CROSS_ENCODER_DEVICE)
        if model is None:
            return None
        return float(model.predict([[query, passage]], show_progress_bar=False)[0])
    except Exception as e:  # noqa: BLE001 — detector must degrade silently
        print(f"[WEB] relatedness scoring failed ({type(e).__name__}: {e}).", flush=True)
        return None


def _rerank_snippets(query, results):
    """Stage 1: reorder DDG results best-first by (title + snippet) relevance.
    Returns [(result, score), ...]; score is None when reranking is unavailable."""
    texts = [f"{r.get('title', '')} {r.get('body', '')}".strip() for r in results]
    scores = _score_relevance(query, texts)
    if scores is None:
        return [(r, None) for r in results]
    order = list(np.argsort(scores)[::-1])
    return [(results[i], scores[i]) for i in order]


def _fetch_pages(results):
    """Fetch + extract page text for several results IN PARALLEL, returning {url: (text, date)}
    for the pages that came back usable (date is 'YYYY-MM-DD' or None — item 4).

    Returns EARLY — as soon as WEB_FETCH_MIN_PAGES usable pages are in AND the stage-1 top
    candidate has resolved — instead of waiting on stragglers. A slow lower-ranked page is
    skipped (observed live: 2/3 back fast, the 3rd held the full timeout for nothing); a slow
    TOP-ranked page is still waited for, so speed never costs us the best candidate. The
    WEB_FETCH_TIMEOUT remains the hard backstop for a genuine hang. `results` is in stage-1
    ranked order, so urls[0] is the top candidate."""
    urls = [r.get("href", "") for r in results if r.get("href")]
    if not urls:
        return {}
    top_url = urls[0]
    target = min(WEB_FETCH_MIN_PAGES, len(urls))
    out, resolved = {}, set()
    ex = _cf.ThreadPoolExecutor(max_workers=len(urls))
    futs = {ex.submit(fetch_page_text, u, with_date=True): u for u in urls}
    try:
        for fut in _cf.as_completed(futs, timeout=WEB_FETCH_TIMEOUT):
            u = futs[fut]
            resolved.add(u)
            try:
                txt, date = fut.result()
            except Exception:  # noqa: BLE001 — a single failed fetch just drops out
                txt, date = "", None
            if txt:
                out[u] = (txt, date)
            # Enough usable pages for a real stage-2 comparison AND the best candidate is in:
            # stop waiting on the rest.
            if len(out) >= target and top_url in resolved:
                if len(resolved) < len(urls):
                    print(f"[WEB] {len(out)} pages in; skipping "
                          f"{len(urls) - len(resolved)} straggler(s).", flush=True)
                break
    except _cf.TimeoutError:
        print(f"[WEB] fetch timeout {WEB_FETCH_TIMEOUT}s; "
              f"{len(out)}/{len(urls)} pages back.", flush=True)
    finally:
        # Don't block on stragglers (a with-block / shutdown(wait=True) would defeat the early
        # return and the timeout).
        ex.shutdown(wait=False, cancel_futures=True)
    return out


def _page_date(downloaded):
    """Publish date 'YYYY-MM-DD' for already-downloaded HTML, or None. Reads trafilatura's
    metadata (htmldate under the hood). An undated page returns None (verified 2026-06-08,
    trafilatura 2.0.0 / htmldate 1.9.4 — extract_metadata does NOT guess a date), so we never
    surface a fabricated date. Item 4 / source-metadata transparency."""
    try:
        md = trafilatura.extract_metadata(downloaded)
        return (getattr(md, "date", None) or None) if md else None
    except Exception:  # noqa: BLE001 — date is best-effort; absence must degrade silently
        return None


def fetch_page_text(url, include_tables=True, with_date=False):
    """Returns extracted page text (str). With with_date=True returns (text, date) where date is
    the publish date 'YYYY-MM-DD' or None — parsed from the SAME download (no extra fetch)."""
    downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        return ("", None) if with_date else ""

    # /read passes include_tables=False: infobox/table markup ("| | |---|") is noise
    # when spoken aloud. The summarisation path keeps the default (tables included).
    text = trafilatura.extract(downloaded, include_tables=include_tables) or ""

    if with_date:
        return text, (_page_date(downloaded) if (text and WEB_SOURCE_DATE_ENABLED) else None)
    return text


def get_web_context(query):
    results = search_web(query, max_results=WEB_RESULTS_FETCH)
    if not results:
        return None

    # Stage 1: rerank DDG snippets, keep the top-K candidates to actually fetch.
    ranked = _rerank_snippets(query, results)
    if WEB_RANK_DEBUG:
        for r, s in ranked:
            tag = f"{s:6.3f}" if s is not None else "   n/a"
            print(f"[WEB-RANK]  {tag}  {r.get('title', '')[:70]} — {r.get('href', '')}",
                  flush=True)
    top = [r for r, _ in ranked[:WEB_FETCH_TOP_K]]

    # Stage 2: fetch those pages in parallel, then re-rank on their ACTUAL extracted text
    # (evidence, not a one-line snippet). Robust: a lower stage-1 page with real content can
    # overtake a #1 whose page is empty/unfetchable.
    pages = _fetch_pages(top)
    have = [(r, td[0], td[1]) for r in top
            for td in [pages.get(r.get("href", ""), ("", None))] if td[0].strip()]

    if have:
        scores = _score_relevance(query, [t[:WEB_CONTENT_RERANK_CHARS] for _, t, _ in have])
        if scores is not None:
            order = list(np.argsort(scores)[::-1])
            have = [have[i] for i in order]
            ordered_scores = [scores[i] for i in order]
            best_score = ordered_scores[0]
            if WEB_RANK_DEBUG:
                for (r, _, _), s in zip(have, ordered_scores):
                    print(f"[WEB-RANK2] {s:6.3f}  {r.get('title', '')[:70]} — "
                          f"{r.get('href', '')}", flush=True)
        else:
            best_score = ranked[0][1]   # stage-1 score (may be None)
        best, best_text, best_date = have[0]
    else:
        # No page text from any candidate — fall back to the stage-1 best snippet.
        best, best_text, best_score, best_date = top[0], "", ranked[0][1], None

    # Empty-retrieval honesty: if even the best content score is below the floor, treat it as
    # no useful source — the router then speaks the honest apology instead of handing the model
    # an off-topic page to confabulate over. Floor 0.0 = disabled until calibrated.
    if best_score is not None and best_score < WEB_RELEVANCE_FLOOR:
        print(f"[WEB] best relevance {best_score:.3f} < floor {WEB_RELEVANCE_FLOOR}; "
              f"no useful source.", flush=True)
        return None

    title = best.get("title", "")
    url = best.get("href", "")
    content = best_text[:6000] if best_text.strip() else best.get("body", "")
    print(f"[WEB] {title} — {url}  (published {best_date or 'n/a'})")
    return {"title": title, "url": url, "content": content, "date": best_date}


# ── /read source fetchers ───────────────────────────────────────────────────
# Each returns {"title", "url", "content"} or None. Content is clean, readable
# prose (no snippets) sized for TTS: Wikipedia uses its bounded summary extract;
# Tavily/Exa full-page text is capped at READ_MAX_CHARS.

def fetch_wikipedia(query, lang="en"):
    """Resolve a fuzzy query to a canonical title (opensearch), then pull the clean
    REST summary extract. Verified live against en.wikipedia.org."""
    base = f"https://{lang}.wikipedia.org"
    title = query
    try:
        r = requests.get(f"{base}/w/api.php",
                         params={"action": "opensearch", "search": query, "limit": 1, "format": "json"},
                         headers=_UA, timeout=8)
        hits = r.json()
        if isinstance(hits, list) and len(hits) > 1 and hits[1]:
            title = hits[1][0]
    except Exception as e:
        print(f"[READ] Wikipedia opensearch failed ({e}); trying raw title.", flush=True)
    try:
        slug = requests.utils.quote(title.replace(" ", "_"), safe="")
        r = requests.get(f"{base}/api/rest_v1/page/summary/{slug}", headers=_UA, timeout=8)
        if r.status_code != 200:
            return None
        d = r.json()
        extract = (d.get("extract") or "").strip()
        if not extract:
            return None
        url = d.get("content_urls", {}).get("desktop", {}).get("page", "")
        return {"title": d.get("title", title), "url": url, "content": extract}
    except Exception as e:
        print(f"[READ] Wikipedia summary failed ({e}).", flush=True)
        return None


def fetch_tavily(query):
    """Tavily search → top result. include_raw_content gives the full page text in
    one call; fall back to trafilatura on the URL if absent. UNVERIFIED LIVE (key
    present, but no credit spent yet) — field shape per Tavily's documented REST API."""
    if not TAVILY_API_KEY:
        print("[READ] Tavily key missing.", flush=True)
        return None
    try:
        r = requests.post("https://api.tavily.com/search",
                          json={"api_key": TAVILY_API_KEY, "query": query, "max_results": 3},
                          headers=_UA, timeout=15)
        if r.status_code != 200:
            print(f"[READ] Tavily HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return None
        results = r.json().get("results") or []
        if not results:
            return None
        best = results[0]
        url = best.get("url", "")
        # Clean prose for TTS: trafilatura on the URL (Tavily's raw_content is dirty
        # markdown — nav/image junk). Fall back to Tavily's clean snippet if extraction fails.
        content = fetch_page_text(url, include_tables=False) or (best.get("content") or "").strip()
        if not content:
            return None
        return {"title": best.get("title", ""), "url": url,
                "content": content[:READ_MAX_CHARS]}
    except Exception as e:
        print(f"[READ] Tavily failed ({e}).", flush=True)
        return None


def fetch_exa(query):
    """Exa semantic search with contents.text → top result page text in one call;
    fall back to trafilatura if absent. UNVERIFIED LIVE (key present, no credit spent
    yet) — field shape per Exa's documented REST API."""
    if not EXA_API_KEY:
        print("[READ] Exa key missing.", flush=True)
        return None
    try:
        r = requests.post("https://api.exa.ai/search",
                          json={"query": query, "numResults": 3, "contents": {"text": True}},
                          headers={**_UA, "x-api-key": EXA_API_KEY}, timeout=15)
        if r.status_code != 200:
            print(f"[READ] Exa HTTP {r.status_code}: {r.text[:200]}", flush=True)
            return None
        results = r.json().get("results") or []
        if not results:
            return None
        best = results[0]
        content = (best.get("text") or "").strip()
        if not content:
            content = fetch_page_text(best.get("url", ""))
        if not content:
            return None
        return {"title": best.get("title", ""), "url": best.get("url", ""),
                "content": content[:READ_MAX_CHARS]}
    except Exception as e:
        print(f"[READ] Exa failed ({e}).", flush=True)
        return None


def answer_from_web(query, summarize_function):
    total_start = time.perf_counter()

    search_start = time.perf_counter()
    results = search_web(query)
    print(f"[TIME] Web search: {time.perf_counter() - search_start:.2f}s")

    if not results:
        return "I could not find anything useful."

    best = results[0]

    title = best.get("title", "Unknown")
    url = best.get("href", "")
    snippet = best.get("body", "")

    print(f"[WEB] Found: {title}")
    print(f"[WEB] URL: {url}")

    fetch_start = time.perf_counter()
    page_text = fetch_page_text(url)
    print(f"[TIME] Page fetch/extract: {time.perf_counter() - fetch_start:.2f}s")

    source_text = page_text[:6000] if page_text else snippet

    prompt = f"""
User question:
{query}

Source title:
{title}

Source URL:
{url}

Source content:
{source_text}

Please summarize this clearly and naturally.
"""

    summary_start = time.perf_counter()
    answer = summarize_function(prompt)
    print(f"[TIME] Summary generation: {time.perf_counter() - summary_start:.2f}s")
    print(f"[WEB] Summary answer: {answer}")
    print(f"[TIME] Total web pipeline: {time.perf_counter() - total_start:.2f}s")

    return answer