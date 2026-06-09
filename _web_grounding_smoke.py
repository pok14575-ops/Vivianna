r"""Offline smoke for the log-only web-answer grounding guard (brain._web_answer_grounding).

Imports the real brain module but STUBS grounding.entailment_prob, so no NLI model / GPU /
server is needed. Exercises the actual confabulation from the 2026-06-08 live run (router
picked the correct wedding article; the 9B invented a date + venue the page never contained)
and checks: questions/filler are skipped, the fabricated sentence is the weakest claim and is
flagged LIKELY-CONFABULATION, and every fallback path (disabled / no evidence / model gone) is
a silent no-op. The real-model pass is Jamie's; this proves the plumbing + claim logic.

Run: F:\AI\Vivianna\venv\Scripts\python.exe F:\AI\Vivianna\_web_grounding_smoke.py
"""
import io
from contextlib import redirect_stdout

import brain
import grounding

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


# The page actually said only "should marry in summer 2026" — no exact date, no venue.
EVIDENCE = ("Taylor Swift und Travis Kelce sollen im Sommer 2026 heiraten. "
            "Die Verlobung wurde im August 2025 bekannt gegeben.")
# Verbatim shape of the live answer: a fabricated-specifics sentence, a derived sentence,
# and a trailing question (must be skipped).
ANSWER = ("Taylor Swift and Travis Kelce are reportedly set to get married on June 13, 2026, "
          "at a luxury resort in Rhode Island called the Ocean House. "
          "The couple announced their engagement back in August 2025. "
          "Since today is June 8, 2026, that would be just five days away! "
          "Would you like to know anything else about the event?")


def fake_entail(evidence, claim, **kw):
    """Simulate the NLI: the page supports the 'summer 2026 marriage' gist, but NOT the
    fabricated specifics (exact date / venue / derived countdown)."""
    low = ["june 13", "ocean house", "rhode island", "five days"]
    return 0.08 if any(m in claim.lower() for m in low) else 0.82


def run():
    buf = io.StringIO()
    with redirect_stdout(buf):
        brain._web_answer_grounding(EVIDENCE, ANSWER)
    return buf.getvalue()


grounding.entailment_prob = fake_entail

# 1. Happy path: fabricated sentence is the weakest claim and is flagged.
out = run()
check("emits per-sentence [WEB-GROUND] lines", "[WEB-GROUND] entail=" in out)
check("flags the fabrication as LIKELY-CONFABULATION", "LIKELY-CONFABULATION" in out)
check("min entailment is the fabricated value (0.080)", "min_entail=0.080" in out)
check("trailing question is skipped (not scored)", "anything else" not in out)
check("the supported-gist threshold tag appears (0.820 not LOW)",
      "entail=0.820" in out and "0.820 LOW" not in out)

# 2. Disabled -> total no-op.
brain.WEB_GROUNDING_CHECK_ENABLED = False
check("WEB_GROUNDING_CHECK_ENABLED=False -> no output", run().strip() == "")
brain.WEB_GROUNDING_CHECK_ENABLED = True

# 3. No evidence -> no-op.
buf = io.StringIO()
with redirect_stdout(buf):
    brain._web_answer_grounding("", ANSWER)
check("empty evidence -> no output", buf.getvalue().strip() == "")

# 4. Model unavailable (entailment_prob None) -> early return, no min line, no crash.
grounding.entailment_prob = lambda *a, **k: None
out = run()
check("model unavailable -> no min_entail line, no crash", "min_entail=" not in out)
grounding.entailment_prob = fake_entail

# 5. Only-questions/filler answer -> nothing to ground, no-op.
buf = io.StringIO()
with redirect_stdout(buf):
    brain._web_answer_grounding(EVIDENCE, "Sure! Okay. Want more?")
check("no factual claims -> no output", buf.getvalue().strip() == "")

# 6. (5a) The HONEST disclaimer answer must NOT be flagged. This is the real 2026-06-08 turn-6
# shape: she correctly said the answer wasn't in the page and proposed a better source. The NLI
# (here: low for everything, mimicking the live 0.092) does NOT entail these sentences — without
# the disclaimer filter they'd be false-flagged LIKELY-CONFABULATION (the exact bug 5a fixes).
HONEST = ("I could not find reliable vitamin C content for strawberries in the search results. "
          "The provided article does not contain nutritional data. "
          "I would need to check a dedicated source like USDA FoodData Central. "
          "Would you like me to try a new search?")
grounding.entailment_prob = lambda evidence, claim, **kw: 0.092  # un-entailed, like the live NLI
buf = io.StringIO()
with redirect_stdout(buf):
    brain._web_answer_grounding(EVIDENCE, HONEST)
out = buf.getvalue()
check("honest disclaimers are skipped (not scored)", "skip (disclaimer)" in out)
check("honest answer is NOT flagged LIKELY-CONFABULATION", "LIKELY-CONFABULATION" not in out)
check("honest answer emits no min_entail (all claims filtered out)", "min_entail=" not in out)

# 7. (5a) MIXED answer: one honest disclaimer + one real fabrication. The disclaimer is skipped,
# but the fabrication MUST still be caught — the filter must not become a confab loophole.
grounding.entailment_prob = fake_entail  # low only for the fabricated specifics
MIXED = ("I could not find the exact wedding date in the search results. "
         "The ceremony will be held at the Ocean House in Rhode Island.")
buf = io.StringIO()
with redirect_stdout(buf):
    brain._web_answer_grounding(EVIDENCE, MIXED)
out = buf.getvalue()
check("mixed: disclaimer sentence skipped", "skip (disclaimer)" in out)
check("mixed: real fabrication still flagged", "LIKELY-CONFABULATION" in out)
check("mixed: disclaimer not the weakest (date sentence not scored)",
      "exact wedding date" not in out.split("min_entail")[-1])

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
