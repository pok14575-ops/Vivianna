r"""Offline smoke for item 2 — referential / meta query repair (router._resolve_search_query +
the route() web-branch integration). No network, no model: nli_classify / get_web_context /
last_user_message / emit are stubbed so we test only the repair logic and that the rebuilt query
is what actually reaches get_web_context. The live pass (real ASR follow-up over a real prior
turn) is Jamie's.

Run: F:\AI\Vivianna\venv\Scripts\python.exe F:\AI\Vivianna\_query_repair_smoke.py
"""
import router

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


VITC = "How much vitamin C is in a strawberry?"

# 1. Referential / meta follow-up + a real prior topic -> rebuilt to the prior topic.
REFERENTIAL_OK = [
    ("look it up on a reliable source", VITC),
    ("tell me more", "Tell me about the Eiffel Tower history"),
    ("search for it", "what is the population of France"),
    ("look it up", "the boiling point of liquid nitrogen"),
    ("on a reliable source", "how many calories are in an avocado"),
    ("verify that", "Einstein was born in 1879 in Germany"),
    ("what about it", "the GDP of Japan"),
    ("from a different website", "the height of Mount Everest"),
    # Broadened after live run 3 (these all FELL THROUGH before, searching the literal text):
    ("Maybe try a more reliable source?", VITC),
    ("Yes take more reputable news please.", VITC),
    ("Please look up on reuters.", VITC),
    ("search the guardian", VITC),
    ("try a different outlet", VITC),
    ("use a credible source", VITC),
    ("any other sources?", VITC),
    ("check bbc instead", VITC),
]
for q, prior in REFERENTIAL_OK:
    check(f"referential {q!r} -> prior topic",
          router._resolve_search_query(q, prior) == prior)

# 2. Standalone queries (carry their own topic) -> pass through unchanged (None).
STANDALONE = [
    ("what is the capital of Brazil", VITC),
    ("look up the population of France", VITC),     # 'look up X', not 'look it up'
    ("who won the 2024 World Series", "x y z topic"),
    ("how much vitamin C is in a strawberry", "earlier chit chat here"),
    # Guards against the broadened matcher over-rewriting standalone queries that merely END
    # in a source-ish word or contain a lookup verb + real content:
    ("tell me the latest sports news", VITC),
    ("get the latest news headlines", VITC),
    ("show me the articles", VITC),
    ("I love reading the news", VITC),
    ("look up the population of France", VITC),
]
for q, prior in STANDALONE:
    check(f"standalone {q!r} -> unchanged (None)",
          router._resolve_search_query(q, prior) is None)

# 3. No usable prior (empty / greeting / itself referential) -> None, even if input is referential.
check("referential but empty prior -> None",
      router._resolve_search_query("look it up", "") is None)
check("referential but one-word prior -> None",
      router._resolve_search_query("look it up", "hi") is None)
check("referential but acknowledgement prior -> None",
      router._resolve_search_query("tell me more", "ok") is None)
check("referential but prior is itself referential -> None",
      router._resolve_search_query("look it up", "search for it") is None)

# ── route() integration: the rebuilt query is what reaches get_web_context ──────
captured = {}
router.nli_classify       = lambda x: ("web", 0.95)
router.role_check         = lambda x: "ok"
router.parse_source_request = lambda x: None
router.emit               = lambda *a, **k: None
router.get_task_acknowledgement = lambda *a, **k: ""
router.detect_language    = lambda x: "en"
router.set_route_confidence = lambda *a, **k: None
def _capture_web(q):
    captured["q"] = q
    return None   # no context -> _do_web takes the apology path; we only care about the query


router.get_web_context = _capture_web


def fake_respond(user_input, **kw):
    captured["respond_input"] = user_input
    return None


def run_route(user_input, prior):
    captured.clear()
    router.last_user_message = lambda: prior
    router.route(user_input, fake_respond)
    return captured


c = run_route("look it up on a reliable source", VITC)
check("route: referential follow-up searches the inherited topic", c.get("q") == VITC)
check("route: user-facing input is still what the user said",
      c.get("respond_input") == "look it up on a reliable source")

c = run_route("what is the boiling point of water", VITC)
check("route: standalone query searches its own text",
      c.get("q") == "what is the boiling point of water")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
