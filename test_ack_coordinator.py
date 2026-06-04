# test_ack_coordinator.py
"""
Frozen regression tests for the Ack Coordinator (Phase A, step 3).

The first test is the one that matters most: it locks the precedence-resolution
rule so the first-come-boolean bug we caught in design can never silently return.
Run:  python test_ack_coordinator.py        (no pytest dependency required)
"""

from __future__ import annotations

import random

from ack_coordinator import (
    AckCategory,
    AckConfig,
    AckCoordinator,
    DeterministicGate,
    GateResult,
    ManualGate,
    run_turn,
    ack_prompt_constraint,
)


def _coord(**cfg_kw) -> AckCoordinator:
    # Recording stub emit + seeded rng for deterministic wording.
    return AckCoordinator(cfg=AckConfig(debug=False, **cfg_kw), rng=random.Random(0))


# ── THE precedence regression (do not delete) ──────────────────────────────────
def test_precedence_safety_beats_emotional_regardless_of_submit_order():
    """Emotional submitted FIRST, safety SECOND. Safety (higher precedence) must
    still win. This is the exact inversion the boolean-flag design would produce."""
    c = _coord()
    c.reset_turn()
    c.submit(AckCategory.EMOTIONAL, confidence=1.0, source="emo")     # evaluated first
    c.submit(AckCategory.SAFETY, confidence=1.0, source="safety")     # evaluated second
    c.resolve_and_emit()
    assert c.last_emitted_category == AckCategory.SAFETY, c.last_emitted_category
    assert len(c._emitted) == 1


def test_full_precedence_ladder():
    """Lowest priority number wins across the whole ladder, any submit order."""
    order = [
        AckCategory.THINKING,
        AckCategory.EMOTIONAL,
        AckCategory.MEMORY_CONFLICT,
        AckCategory.CRITICAL_TECHNICAL,
        AckCategory.SAFETY,
        AckCategory.TOOL_TASK,   # should win
    ]
    c = _coord()
    c.reset_turn()
    for cat in order:
        c.submit(cat, confidence=1.0, source=cat)
    c.resolve_and_emit()
    assert c.last_emitted_category == AckCategory.TOOL_TASK, c.last_emitted_category


# ── one-ack-per-turn ───────────────────────────────────────────────────────────
def test_one_ack_per_turn():
    c = _coord()
    c.reset_turn()
    c.submit(AckCategory.SAFETY, confidence=1.0, source="a")
    first = c.resolve_and_emit()
    # A second resolve in the same turn must be a no-op, even with new candidates.
    c.submit(AckCategory.TOOL_TASK, confidence=1.0, source="b")
    second = c.resolve_and_emit()
    assert first is not None
    assert second is None
    assert len(c._emitted) == 1


def test_reset_turn_allows_next_ack():
    c = _coord()
    c.reset_turn()
    c.submit(AckCategory.SAFETY, confidence=1.0, source="a")
    c.resolve_and_emit()
    c.reset_turn()
    c.submit(AckCategory.EMOTIONAL, confidence=1.0, source="b")
    c.resolve_and_emit()
    assert len(c._emitted) == 2
    assert c.last_emitted_category == AckCategory.EMOTIONAL


# ── no candidates -> silence ────────────────────────────────────────────────────
def test_no_candidates_no_ack():
    c = _coord()
    c.reset_turn()
    assert c.resolve_and_emit() is None
    assert c._emitted == []
    assert c.prompt_constraint() == ""


# ── neutral fallback on low confidence ──────────────────────────────────────────
def test_low_confidence_falls_back_to_neutral():
    c = _coord(neutral_fallback_threshold=0.75)
    c.reset_turn()
    # Emotional category but the gate is unsure -> must NOT emit an emotional-toned
    # ack; commit to neutral instead (wrong tone is irreversible).
    c.submit(AckCategory.EMOTIONAL, confidence=0.40, source="unsure")
    c.resolve_and_emit()
    assert c.last_emitted_category == AckCategory.NEUTRAL, c.last_emitted_category


def test_high_confidence_keeps_category():
    c = _coord(neutral_fallback_threshold=0.75)
    c.reset_turn()
    c.submit(AckCategory.EMOTIONAL, confidence=0.95, source="sure")
    c.resolve_and_emit()
    assert c.last_emitted_category == AckCategory.EMOTIONAL


# ── prompt feed-forward is a real promise ──────────────────────────────────────
def test_prompt_constraint_follows_committed_category():
    c = _coord()
    c.reset_turn()
    c.submit(AckCategory.EMOTIONAL, confidence=1.0, source="emo")
    c.resolve_and_emit()
    constraint = c.prompt_constraint()
    assert "gently" in constraint.lower()
    # Neutral commit yields no constraint line.
    assert ack_prompt_constraint(AckCategory.NEUTRAL) == ""


# ── run_turn() orchestration shape ─────────────────────────────────────────────
def test_run_turn_emits_from_gate():
    c = _coord()
    gate = ManualGate(GateResult(should_ack=True, category=AckCategory.CRITICAL_TECHNICAL,
                                 confidence=1.0, source="manual"))
    text = run_turn(c, gate, "explain the KV cache quant tradeoff")
    assert text is not None
    assert c.last_emitted_category == AckCategory.CRITICAL_TECHNICAL


def test_run_turn_gate_declines():
    c = _coord()
    gate = ManualGate(GateResult(should_ack=False))
    text = run_turn(c, gate, "what time is it")
    assert text is None
    assert c._emitted == []


# ── DeterministicGate (Phase A authority) ──────────────────────────────────────
def test_gate_routine_input_no_ack():
    g = DeterministicGate()
    for routine in ["what time is it", "summarise this article", "how do i sort a list in python",
                    "thanks, that helped"]:
        v = g.evaluate(routine)
        assert v.should_ack is False, routine


def test_gate_emotional_fires_emotional():
    g = DeterministicGate()
    v = g.evaluate("i can't sleep and i feel so low tonight")
    assert v.should_ack and v.category == AckCategory.EMOTIONAL, v


def test_gate_safety_outranks_when_both_present():
    g = DeterministicGate()
    # Distress (emotional) AND self-harm (safety) — safety must win.
    v = g.evaluate("i'm so overwhelmed i want to hurt myself")
    assert v.should_ack and v.category == AckCategory.SAFETY, v


def test_gate_memory_assertion_is_anticipatory():
    g = DeterministicGate()
    v = g.evaluate("but you told me you would remember my sister's name")
    assert v.should_ack and v.category == AckCategory.MEMORY_CONFLICT, v


def test_gate_critical_tech():
    g = DeterministicGate()
    v = g.evaluate("just run rm -rf on the whole data folder")
    assert v.should_ack and v.category == AckCategory.CRITICAL_TECHNICAL, v


def test_gate_german_emotional():
    g = DeterministicGate()
    v = g.evaluate("ich habe angst und bin total überfordert")
    assert v.should_ack and v.category == AckCategory.EMOTIONAL, v


def test_gate_drives_coordinator_end_to_end():
    c = _coord()
    text = run_turn(c, DeterministicGate(), "i'm scared about the surgery tomorrow")
    assert text is not None
    assert c.last_emitted_category == AckCategory.EMOTIONAL
    assert "gently" in c.prompt_constraint().lower()


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} passed.")


if __name__ == "__main__":
    _run_all()
