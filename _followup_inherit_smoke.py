r"""Offline smoke for item 1 — web follow-up inheritance (router._maybe_web_followup +
_do_followup_inherit + route() integration + runtime_state web cache). No network, no model:
score_relatedness / nli_classify / emit etc. are stubbed; the real runtime_state cache (a plain
Python var) is used. Verifies: referential cue acts now, content path is log-only until the
floor is lowered, stale cache is dropped, re-grounding passes the cached article as evidence with
the honesty prompt, and a no-cache chat turn is untouched. Live pass (real ettin relatedness +
floor calibration) is Jamie's.

Run: F:\AI\Vivianna\venv\Scripts\python.exe F:\AI\Vivianna\_followup_inherit_smoke.py
"""
import router
import runtime_state

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


CACHE_ARGS = ("Taylor Swift wedding", "TS Wedding — ZDF",
              "https://zdf.example/ts", "The wedding is planned for summer 2026.")


def set_cache():
    runtime_state.set_web_cache(*CACHE_ARGS)


# Pin item-1 config on the router namespace (it imported the names by value).
router.WEB_FOLLOWUP_ENABLED = True
router.WEB_FOLLOWUP_MAX_AGE_S = 300
router.WEB_FOLLOWUP_RELATEDNESS_FLOOR = 999.0   # default: content path inert (log-only)
router.WEB_FOLLOWUP_DEBUG = True

# ── _maybe_web_followup ───────────────────────────────────────────────────────
runtime_state.clear_web_cache()
router.score_relatedness = lambda q, p: 5.0
check("no cache -> None", router._maybe_web_followup("tell me more") is None)

set_cache()
check("fresh cache + REFERENTIAL cue acts (even with floor inert)",
      router._maybe_web_followup("tell me more") == runtime_state.get_web_cache())

set_cache()
check("fresh cache + CONTENT follow-up below inert floor -> None (log-only)",
      router._maybe_web_followup("any speculation on the locations?") is None)

set_cache()
router.WEB_FOLLOWUP_RELATEDNESS_FLOOR = 3.0      # calibrated: content path now live
check("content follow-up above calibrated floor -> inherits",
      router._maybe_web_followup("any speculation on the locations?") is not None)
router.score_relatedness = lambda q, p: 1.0
set_cache()
check("content follow-up below calibrated floor -> None",
      router._maybe_web_followup("what's the weather tomorrow?") is None)
router.WEB_FOLLOWUP_RELATEDNESS_FLOOR = 999.0
router.score_relatedness = lambda q, p: 5.0

set_cache()
router.WEB_FOLLOWUP_MAX_AGE_S = -1               # force every cache stale
check("stale cache -> None", router._maybe_web_followup("tell me more") is None)
check("stale cache is cleared", runtime_state.get_web_cache() is None)
router.WEB_FOLLOWUP_MAX_AGE_S = 300

set_cache()
router.WEB_FOLLOWUP_ENABLED = False
check("disabled -> None", router._maybe_web_followup("tell me more") is None)
router.WEB_FOLLOWUP_ENABLED = True

# relatedness model unavailable -> referential still acts, content path can't
runtime_state.clear_web_cache(); set_cache()
router.score_relatedness = lambda q, p: None
check("relatedness None: referential still acts", router._maybe_web_followup("tell me more"))
set_cache()
check("relatedness None: content follow-up -> None",
      router._maybe_web_followup("any speculation on locations?") is None)
router.score_relatedness = lambda q, p: 5.0

# ── _do_followup_inherit ──────────────────────────────────────────────────────
router.detect_language = lambda x: "en"
captured = {}


def fake_respond(user_input, **kw):
    captured.clear()
    captured["user_input"] = user_input
    captured.update(kw)
    return "RESP"


cache = runtime_state.get_web_cache()
out = router._do_followup_inherit("speculations on locations?", fake_respond, cache)
check("inherit returns respond() result", out == "RESP")
check("inherit grounds over the cached content",
      captured.get("grounding_evidence") == CACHE_ARGS[3])
check("inherit is tool_driven (kept out of LTM)", captured.get("tool_driven") is True)
check("inherit prompt embeds the cached article + honesty instruction",
      CACHE_ARGS[3] in captured.get("llm_input", "")
      and "rather than guessing or speculating" in captured.get("llm_input", ""))
check("inherit keeps the user's literal question as user_input",
      captured.get("user_input") == "speculations on locations?")

# ── route() integration ───────────────────────────────────────────────────────
router.role_check = lambda x: "ok"
router.parse_source_request = lambda x: None
router.emit = lambda *a, **k: None
router.get_task_acknowledgement = lambda *a, **k: ""
router.set_route_confidence = lambda *a, **k: None
router.last_user_message = lambda: ""
router.nli_classify = lambda x: ("chat", 0.80)   # plain chat route (not web, not low-conf)

runtime_state.clear_web_cache(); set_cache()
router.route("tell me more", fake_respond)
check("route: referential follow-up re-grounds over cache (evidence attached)",
      captured.get("grounding_evidence") == CACHE_ARGS[3])

runtime_state.clear_web_cache()
captured.clear()
router.route("what is your favourite colour", fake_respond)
check("route: no cache -> ordinary chat (no grounding evidence)",
      "grounding_evidence" not in captured and captured.get("user_input") == "what is your favourite colour")

# ── Item 4b: provenance question answered from cached source metadata ──────────
for q in ["what's your source?", "Drom what date is your source ?", "where did you get that?",
          "when was that published?", "how reliable is that source?", "what date is the source"]:
    check(f"provenance Q detected: {q!r}", router._is_provenance_question(q))
for q in ["tell me more", "who is Travis", "what is the weather tomorrow", "thanks"]:
    check(f"not a provenance Q: {q!r}", not router._is_provenance_question(q))

runtime_state.clear_web_cache()
check("no cache -> _maybe_provenance None", router._maybe_provenance("what's your source?") is None)
runtime_state.set_web_cache("Taylor Swift wedding", "TS Wedding", "https://x.example/ts",
                            "body text", date="2026-04-07")
check("cache + provenance Q -> returns cache",
      router._maybe_provenance("what's your source?") is not None)
check("cache + non-provenance Q -> None", router._maybe_provenance("tell me more") is None)

captured.clear()
router.route("Drom what date is your source ?", fake_respond)   # 'what date' still matches
ev = captured.get("grounding_evidence") or ""
check("route: provenance answered from cache (url in grounding evidence)", "https://x.example/ts" in ev)
check("route: provenance evidence carries the publish date", "2026-04-07" in ev)
check("route: provenance prompt forbids inventing (tool_driven, not stored)",
      captured.get("tool_driven") is True)
runtime_state.clear_web_cache()

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
