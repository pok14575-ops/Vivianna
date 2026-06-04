from ddgs import DDGS
import trafilatura
import time
import requests
from config import TAVILY_API_KEY, EXA_API_KEY, READ_MAX_CHARS

_UA = {"User-Agent": "Vivianna/1.0 (local assistant)"}

def search_web(query, max_results=3):
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    return results


def fetch_page_text(url, include_tables=True):
    downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        return ""

    # /read passes include_tables=False: infobox/table markup ("| | |---|") is noise
    # when spoken aloud. The summarisation path keeps the default (tables included).
    text = trafilatura.extract(downloaded, include_tables=include_tables)

    return text if text else ""


def get_web_context(query):
    results = search_web(query)
    if not results:
        return None
    best = results[0]
    title = best.get("title", "")
    url = best.get("href", "")
    snippet = best.get("body", "")
    page_text = fetch_page_text(url)
    content = page_text[:6000] if page_text else snippet
    print(f"[WEB] {title} — {url}")
    return {"title": title, "url": url, "content": content}


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