r"""Offline smoke for [new-3] B1 — the SUGGEST_SEARCH marker plumbing (brain._suppress_suggest +
_extract_suggestion). No model: tests the pure parse/suppression helpers and a token-by-token
stream simulation proving the marker NEVER leaks to console/TTS while the suggested query is
captured. (B2 = arm/confirm/execute.)

Run: F:\AI\Vivianna\venv\Scripts\python.exe F:\AI\Vivianna\_research_clarify_smoke.py
"""
import brain
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


# ── _extract_suggestion ───────────────────────────────────────────────────────
a, s = brain._extract_suggestion("Just a normal answer with no marker.")
check("no marker -> (text, None)", a == "Just a normal answer with no marker." and s is None)

full = ("I couldn't find it in The Guardian. Would you like me to try AP News?\n"
        "SUGGEST_SEARCH: Taylor Swift wedding date AP News")
a, s = brain._extract_suggestion(full)
check("marker -> answer stripped of marker line",
      a == "I couldn't find it in The Guardian. Would you like me to try AP News?"
      and "SUGGEST_SEARCH" not in a)
check("marker -> suggested query captured", s == "Taylor Swift wedding date AP News")

a, s = brain._extract_suggestion("Answer.\nSUGGEST_SEARCH: q1\nignored trailing line")
check("suggestion is only the first line after the marker", s == "q1")

a, s = brain._extract_suggestion("Answer here.\nSUGGEST_SEARCH:   ")
check("empty suggestion -> None", s is None and a == "Answer here.")

a, s = brain._extract_suggestion('Answer.\nSUGGEST_SEARCH: "Jay Chou biography"')
check("wrapping quotes stripped from suggestion (live run 5)", s == "Jay Chou biography")

# ── _suppress_suggest (streaming predicate) ──────────────────────────────────
check("full marker present -> suppress", brain._suppress_suggest("blah\nSUGGEST_SEARCH: x"))
check("tail 'SUG' after newline -> suppress", brain._suppress_suggest("answer.\nSUG"))
check("tail 'S' (too short) -> no suppress", not brain._suppress_suggest("answer.\nS"))
check("normal prose -> no suppress", not brain._suppress_suggest("The answer is probably yes"))
check("newline mid-answer, normal next line -> no suppress",
      not brain._suppress_suggest("Para one.\nPara two begins"))

# ── stream simulation: marker must never leak; suggestion captured ────────────
def simulate(tokens):
    full, shown = "", ""
    for tok in tokens:
        full += tok
        if brain._suppress_suggest(full):
            continue
        shown += tok
    return full, shown

ANSWER = "I couldn't find it here. Want me to try AP News?"
TOKENS = ["I couldn't", " find it", " here.", " Want me", " to try", " AP News?",
          "\n", "SUGGEST", "_SEARCH", ":", " Taylor Swift", " wedding AP"]
full, shown = simulate(TOKENS)
check("stream: marker text never reaches the shown (spoken/console) output",
      "SUGGEST" not in shown)
check("stream: shown output equals the clean answer", shown.strip() == ANSWER)
ans, sugg = brain._extract_suggestion(full)
check("stream: suggestion parsed from full text", sugg == "Taylor Swift wedding AP")

# ── B2: _is_affirmative ───────────────────────────────────────────────────────
for t in ["yes", "Sure go ahead.", "yes please try any of them", "ok", "go ahead", "please do",
          "Yes take more reputable news please"]:
    check(f"affirmative: {t!r}", router._is_affirmative(t))
for t in ["no thanks", "nah", "actually what's the weather?", "", "maybe later", "stop"]:
    check(f"not affirmative: {t!r}", not router._is_affirmative(t))

# ── B2: resolve_research (confirm + execute + loop cap) ───────────────────────
router.get_web_context = lambda q: {"title": "T", "url": "u", "content": "c"}
router.emit = lambda *a, **k: None
router.get_task_acknowledgement = lambda *a, **k: ""
router.detect_language = lambda x: "en"
cap = {}


def fake_respond(ui, **kw):
    cap["ui"] = ui
    cap.update(kw)
    return None


# Arming must SPEAK the offer (live run 5: it armed silently — only a debug-watcher could say
# yes). Capture the emitted offer instead of letting it print.
offers = []
brain.tts_emit = lambda text, **k: offers.append(text)
check("offer text contains the query", "Jay Chou biography" in brain._research_offer_text("Jay Chou biography"))
check("offer text is a question", brain._research_offer_text("x").strip().endswith("?"))

brain.clear_pending_research()
check("no pending -> resolve returns False", router.resolve_research("yes", fake_respond) is False)

offers.clear()
brain._arm_research("Taylor Swift wedding AP News")
check("armed -> pending set", brain.pending_research() == {"query": "Taylor Swift wedding AP News"})
check("arming SPEAKS the offer (not silent)",
      len(offers) == 1 and "Taylor Swift wedding AP News" in offers[0] and offers[0].strip().endswith("?"))
out = router.resolve_research("Yes please try any of them.", fake_respond)
check("affirmative -> consumed (True)", out is True)
check("executed the SUGGESTED query (not the literal 'yes')",
      cap.get("ui") == "Taylor Swift wedding AP News")
check("loop cap: re-search runs with allow_research_arm=False",
      cap.get("allow_research_arm") is False)
check("pending cleared after execute", brain.pending_research() is None)

brain._arm_research("q")
check("negative reply -> not consumed (False)", router.resolve_research("no thanks", fake_respond) is False)
check("pending dropped on negative", brain.pending_research() is None)

brain._arm_research("q2")
check("slash command -> not consumed", router.resolve_research("/debug", fake_respond) is False)
check("pending KEPT after slash command", brain.pending_research() == {"query": "q2"})
brain.clear_pending_research()

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
