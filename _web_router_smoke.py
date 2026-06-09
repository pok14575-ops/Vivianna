r"""Offline smoke for the two-stage Smart Search Router (source selection + honesty).

Runs with NO network and NO real model: stubs search_web / fetch_page_text and the shared
cross-encoder with a keyword-overlap scorer. Verifies the *mechanics* the live path relies on
— stage-1 snippet rerank, parallel fetch, stage-2 CONTENT rerank, content-empty robustness,
fallback when the model is gone, and the honesty floor. The live pass (real ettin scores +
real DDG/network + measured latency) is Jamie's; this just proves the plumbing is correct.

Run: F:\AI\Vivianna\venv\Scripts\python.exe F:\AI\Vivianna\_web_router_smoke.py
"""
import time

import web_search as ws
import cross_encoder_model

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


# DDG-shaped results: a Reddit opinion thread sits at index 0 (what the old blind top-1 would
# have picked); an actual forecast page sits lower. SNIPPETS (title+body) are deliberately
# thin so stage-1 alone is imperfect — stage-2 content is what nails it.
RESULTS = [
    {"title": "Is the weather always this bad? : r/berlin",
     "href": "https://reddit.com/x", "body": "honestly i hate the rain lol"},
    {"title": "Berlin 7-day Weather Forecast",
     "href": "https://weather.example/berlin", "body": "forecast for the week"},
    {"title": "Berlin travel guide",
     "href": "https://travel.example/berlin", "body": "things to do in berlin"},
]
# Full extracted page text (what stage-2 actually scores).
PAGES = {
    "https://reddit.com/x": "honestly i hate the rain here lol berlin sucks sometimes",
    "https://weather.example/berlin":
        "berlin weather forecast 7 day temperature rain wind detailed outlook for the week",
    "https://travel.example/berlin": "things to do in berlin museums food nightlife",
}
QUERY = "berlin weather forecast 7 day"
# Item 4: publish dates per page (None = undated). fetch_page_text(with_date=True) returns these.
DATES = {
    "https://weather.example/berlin": "2026-06-01",
    "https://travel.example/berlin": None,
    "https://reddit.com/x": None,
}


def _stub_fetch(pages):
    """Build a fetch_page_text stub honoring the with_date contract: text, or (text, date)."""
    def f(url, include_tables=True, with_date=False):
        txt = pages.get(url, "")
        return (txt, DATES.get(url)) if with_date else txt
    return f


class _StubModel:
    """Scores a [query, passage] pair by case-insensitive token overlap — enough to mimic the
    real reranker: the on-topic forecast outranks the opinion thread once content is in play."""
    def predict(self, pairs, show_progress_bar=False):
        out = []
        for q, p in pairs:
            qt, pt = set(q.lower().split()), set(p.lower().split())
            out.append(float(len(qt & pt)))
        return out


def use_model(model):
    cross_encoder_model.get_model = lambda *a, **k: model


# Pin config to small, deterministic values for the test.
ws.WEB_RERANK_ENABLED = True
ws.WEB_FETCH_TOP_K = 3
ws.WEB_FETCH_MIN_PAGES = 2
ws.WEB_FETCH_TIMEOUT = 5.0
ws.WEB_CONTENT_RERANK_CHARS = 2000
ws.WEB_RELEVANCE_FLOOR = 0.0
TOP_URL = RESULTS[0]["href"]   # stage-1 ordering is preserved by the stub; reddit sits at 0

# ── stage helpers ────────────────────────────────────────────────────────────
use_model(_StubModel())
check("_score_relevance returns per-text scores",
      ws._score_relevance(QUERY, ["berlin weather forecast", "cat pictures"]) == [3.0, 0.0])

use_model(None)
check("_score_relevance None when model unavailable",
      ws._score_relevance(QUERY, ["a b", "c d"]) is None)
use_model(_StubModel())
check("_score_relevance None for <2 texts", ws._score_relevance(QUERY, ["only one"]) is None)

ranked = ws._rerank_snippets(QUERY, RESULTS)
check("stage-1 reorders best-first (forecast above reddit)",
      ranked[0][0]["title"].startswith("Berlin 7-day"))

ws.fetch_page_text = _stub_fetch(PAGES)
fetched = ws._fetch_pages(RESULTS)
check("_fetch_pages returns >= MIN_PAGES usable, incl. the top candidate",
      len(fetched) >= ws.WEB_FETCH_MIN_PAGES and TOP_URL in fetched and all(fetched.values()))

ws.fetch_page_text = lambda url, include_tables=True, with_date=False: (_ for _ in ()).throw(RuntimeError("boom"))
check("_fetch_pages drops failed fetches, never raises", ws._fetch_pages(RESULTS) == {})


def _slow_for(slow_url, secs):
    def f(url, include_tables=True, with_date=False):
        if url == slow_url:
            time.sleep(secs)
        txt = PAGES.get(url, "")
        return (txt, DATES.get(url)) if with_date else txt
    return f

# Straggler skip: a slow NON-top page must not hold the call — return fast with the others.
ws.fetch_page_text = _slow_for("https://travel.example/berlin", 2.0)
t0 = time.perf_counter()
fetched = ws._fetch_pages(RESULTS)
check("slow non-top straggler is skipped (returns well before its 2s)",
      time.perf_counter() - t0 < 1.0 and len(fetched) >= ws.WEB_FETCH_MIN_PAGES)

# Top candidate is waited for even when it's the slow one — speed never drops the best page.
ws.fetch_page_text = _slow_for(TOP_URL, 0.4)
check("slow TOP candidate is still waited for (included)", TOP_URL in ws._fetch_pages(RESULTS))

ws.fetch_page_text = _stub_fetch(PAGES)

# ── full two-stage get_web_context ──────────────────────────────────────────
ws.search_web = lambda q, max_results=3: RESULTS
ws.fetch_page_text = _stub_fetch(PAGES)

ctx = ws.get_web_context(QUERY)
check("stage-2 content rerank picks the forecast page", ctx and "weather" in ctx["url"])
check("returned content is the fetched page text, not the snippet",
      ctx and ctx["content"].startswith("berlin weather forecast"))
check("item 4: publish date of the winning page threads through (2026-06-01)",
      ctx and ctx.get("date") == "2026-06-01")

# Robustness: stage-1 #1 page is empty -> a lower page with real text must still win.
EMPTY_TOP = dict(PAGES, **{"https://weather.example/berlin": ""})
ws.fetch_page_text = _stub_fetch(EMPTY_TOP)
ctx = ws.get_web_context(QUERY)
check("empty top page -> falls through to a page that has content",
      ctx is not None and ctx["content"].strip() and "weather.example" not in ctx["url"])

# Honesty floor: best content score below the floor -> no useful source (router apologizes).
ws.fetch_page_text = _stub_fetch(PAGES)
ws.WEB_RELEVANCE_FLOOR = 999.0
check("below floor -> get_web_context returns None", ws.get_web_context(QUERY) is None)
ws.WEB_RELEVANCE_FLOOR = 0.0

# Model gone entirely -> still returns a usable context (degrades to DDG order + fetched text).
use_model(None)
ctx = ws.get_web_context(QUERY)
check("model unavailable -> still returns a context (fallback)",
      ctx is not None and ctx["content"].strip())

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
