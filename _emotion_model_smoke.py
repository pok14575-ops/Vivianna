r"""Offline smoke for the go_emotions affective-tone wiring (emotion_layer side, no model).

Tests the PURE mapping + single-primary competition: _judge_model(28 probs -> candidates) and
pre_turn(model_scores=...) competing with the regex/architecture signals. emotion_layer.py
imports nothing from the project, so this runs instantly with fake score dicts.

Run:  venv_tf5\Scripts\python.exe _emotion_model_smoke.py   (or any python)
"""
import sys
from emotion_layer import EmotionLayer, EmotionConfig

PASS, FAIL = [], []
def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def fresh(**kw):
    return EmotionLayer(EmotionConfig(debug=False, **kw))


print("\nPart A — _judge_model mapping (threshold 0.30, scale 1.0)")
L = fresh()

# helper: dict with all-zero defaults plus overrides
def scores(**ov):
    base = {lbl: 0.0 for lbl in (
        "gratitude", "love", "caring", "curiosity", "confusion", "fear", "nervousness",
        "neutral", "grief", "pride", "relief", "joy", "anger")}
    base.update(ov)
    return base

c = L._judge_model(scores(gratitude=0.91))
check("gratitude -> warmth candidate", c == [("warmth", 0.91, "go:warmth", 3)])

c = L._judge_model(scores(fear=0.72))
check("fear -> uncertainty candidate", c == [("uncertainty", 0.72, "go:uncertainty", 2)])

c = L._judge_model(scores(confusion=0.55))
check("confusion -> curiosity candidate", c == [("curiosity", 0.55, "go:curiosity", 2)])

c = L._judge_model(scores(love=0.80, gratitude=0.40))
check("facet uses MAX label (love 0.80 over gratitude 0.40)", c == [("warmth", 0.80, "go:warmth", 3)])

c = L._judge_model(scores(gratitude=0.29))
check("below threshold -> no candidate", c == [])

c = L._judge_model(scores(neutral=0.98))
check("dominant neutral ignored (unmapped) -> no candidate", c == [])

c = L._judge_model(scores(grief=0.9, pride=0.9, relief=0.9))
check("dead facets (grief/pride/relief) unmapped -> no candidate", c == [])

c = L._judge_model(scores(gratitude=0.95, fear=0.85, confusion=0.60))
names = sorted(x[0] for x in c)
check("multiple facets co-fire -> all candidates emitted", names == ["curiosity", "uncertainty", "warmth"])

check("None scores -> []", L._judge_model(None) == [])
check("empty scores -> []", L._judge_model({}) == [])

# intensity scale + cap
L2 = fresh(go_emotions_intensity_scale=2.0)
c = L2._judge_model(scores(gratitude=0.80))
check("scale applied + capped at 1.0", c == [("warmth", 1.0, "go:warmth", 3)])

L3 = fresh(go_emotions_threshold=0.05)
c = L3._judge_model(scores(curiosity=0.06))
check("low threshold lets weak signal through", c == [("curiosity", 0.06, "go:curiosity", 2)])


print("\nPart B — pre_turn single-primary competition")

# model-only warmth, no regex
L = fresh()
st = L.pre_turn("(no regex trigger here)", model_scores=scores(gratitude=0.90))
check("model warmth adopted (no regex)", st.primary_state == "warmth" and st.source == "go:warmth")

# model None -> regex-only behaviour unchanged (gratitude regex still fires)
L = fresh()
st = L.pre_turn("thank you so much!", model_scores=None)
check("model None -> regex path intact (gratitude->warmth)",
      st.primary_state == "warmth" and st.source == "gratitude")

# architecture conflict (0.55) vs weak model warmth (0.40) -> conflict wins
L = fresh()
st = L.pre_turn("ok", conflict=True, model_scores=scores(gratitude=0.40))
check("architecture conflict beats weak model warmth", st.primary_state == "uncertainty"
      and st.source == "memory_conflict")

# strong model warmth (0.90) vs conflict (0.55) -> model wins (honest: confident emotion leads)
L = fresh()
st = L.pre_turn("ok", conflict=True, model_scores=scores(gratitude=0.90))
check("strong model warmth beats conflict", st.primary_state == "warmth")

# model catches what regex misses: implicit fear, no keyword
L = fresh()
st = L.pre_turn("I'm really scared about my daughter's surgery tomorrow",
                model_scores=scores(fear=0.88, nervousness=0.70))
check("model catches implicit uncertainty regex would miss", st.primary_state == "uncertainty"
      and st.source == "go:uncertainty")

# neutral-only model scores + no regex -> stays neutral
L = fresh()
st = L.pre_turn("the meeting is at 3pm", model_scores=scores(neutral=0.95))
check("neutral-only -> stays neutral", st.primary_state == "neutral")

# system_cue reflects the model-driven state
L = fresh()
L.pre_turn("x", model_scores=scores(curiosity=0.75))
check("system_cue non-empty for model-driven curiosity", "curiosity" in L.system_cue().lower())


print(f"\n{'='*50}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:", FAIL)
    sys.exit(1)
print("ALL GREEN")
