# ack_coordinator.py
"""
Vivianna — Deterministic Acknowledgement Coordinator  (DRAFT / not yet wired)

Implements the input-side acknowledgement system from the Tiered Cognition + Ack
doctrine (memory: tiered-cognition-ack-doctrine, day 11).

Pipeline position:
    user input
      -> cheap whole-turn gate / sensitivity + CATEGORY classifier
      -> [this module] optional immediate ack       <-- buys conversational time
      -> large judgment tier (only after the ack claim)
      -> Qwen deliberation
      -> answer

Doctrine rules honoured here (verbatim intent):
  - Input-side trigger BUYS TIME; output-side judgment is validation only.
  - ONE ack per turn, max.
  - Precedence is resolved BEFORE emission (collect-then-choose). A first-come
    boolean flag does NOT enforce precedence, because execution order != priority
    order and an ack cannot be un-spoken once emitted.
  - Wording is conditioned on the detected category, else it becomes a robotic tic.
  - Low category confidence -> NEUTRAL fallback (a wrong tone is irreversible and
    worse than generic).
  - "memory_conflict" wording is ANTICIPATORY: a true conflict is only discovered
    post-retrieval (downstream), which breaks input-side timing — so it fires on
    the input *asserting a fact about a remembered entity*, not on confirmed conflict.
  - The chosen category is fed forward into Qwen's prompt as a style constraint:
    the ack is a promise the answer must honour.

This module is intentionally PURE: it imports nothing from the rest of the project.
`emit_fn` is injected (defaults to a recording stub) so the coordinator stays inert
and unit-testable until brain.py wires `output_bus.emit` in. Protected budget:
    input received -> ack audible : target ~300-700 ms
=> the gate handed to this coordinator MUST be cheap; the large judgment tier runs
AFTER resolve_and_emit(), never before.

Phase map (see plan):
  Phase A (steps 1-6): mode="deterministic". Deterministic/Manual gate has authority.
                       Encoder gate, if present, runs in shadow (logged, no authority).
  Phase B (step 7):    mode="shadow". Observe + label; encoder gate still no authority.
  Phase C (steps 8-9): mode="live". Encoder gate controls wording; judgment escalation on.
Promotions are FLAG FLIPS (AckConfig.mode / judgment_enabled), reversible on a bad day.
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Callable, Dict, List, Optional


# ── Categories & precedence ────────────────────────────────────────────────────
# Wording category. Drives which phrase pool is drawn from.
class AckCategory:
    TOOL_TASK = "tool_task"
    SAFETY = "safety"                 # safety / child boundary
    CRITICAL_TECHNICAL = "critical_technical"
    MEMORY_CONFLICT = "memory_conflict"   # anticipatory (see doctrine)
    EMOTIONAL = "emotional"           # sensitive / emotional concern
    THINKING = "thinking"             # generic "let me think"
    NEUTRAL = "neutral"               # confidence-fallback wording ONLY


# Precedence: LOWER number wins (min). Mirrors the doctrine's ranked list:
#   1 explicit tool/task  2 safety/critical  3 emotional/sensitive  4 thinking.
# MEMORY_CONFLICT was not ranked in the original list; placed as a correctness
# concern just under safety/critical.  # CONFIRM rank with Jamie.
class AckPriority(IntEnum):
    TOOL_TASK = 1
    SAFETY = 2
    CRITICAL_TECHNICAL = 2          # same tier as safety (both "critical")
    MEMORY_CONFLICT = 3             # CONFIRM
    EMOTIONAL = 4
    THINKING = 5


_CATEGORY_PRIORITY: Dict[str, AckPriority] = {
    AckCategory.TOOL_TASK: AckPriority.TOOL_TASK,
    AckCategory.SAFETY: AckPriority.SAFETY,
    AckCategory.CRITICAL_TECHNICAL: AckPriority.CRITICAL_TECHNICAL,
    AckCategory.MEMORY_CONFLICT: AckPriority.MEMORY_CONFLICT,
    AckCategory.EMOTIONAL: AckPriority.EMOTIONAL,
    AckCategory.THINKING: AckPriority.THINKING,
}


# ── Wording pools (varied, to avoid the tic) ───────────────────────────────────
# Multiple variants per category; one is chosen per emission. NEUTRAL is the
# safe-default used when category confidence is below threshold.
_ACK_POOLS: Dict[str, List[str]] = {
    AckCategory.TOOL_TASK: [
        "On it — give me a moment.",
        "Okay, let me pull that up.",
        "Working on it now.",
    ],
    AckCategory.SAFETY: [
        "I want to be careful with this.",
        "Let me take this one carefully.",
        "I want to get this right — give me a second.",
    ],
    AckCategory.CRITICAL_TECHNICAL: [
        "Let me get this right.",
        "Give me a moment to be precise about this.",
        "I want to be exact here — one second.",
    ],
    AckCategory.MEMORY_CONFLICT: [
        "I need to check that against what I remember.",
        "Let me line that up with what I have.",
        "One second — I want to square that with what I know.",
    ],
    AckCategory.EMOTIONAL: [
        "That's a lot — give me a moment.",
        "Give me a second — I want to answer gently.",
        "Take a breath with me — I'm thinking about this.",
    ],
    AckCategory.THINKING: [
        "Let me think about that for a second.",
        "Hmm — give me a moment.",
        "One second while I think this through.",
    ],
    AckCategory.NEUTRAL: [
        "Give me a moment.",
        "One second.",
        "Let me think for a moment.",
    ],
}


# Forward-fed prompt constraint: the ack is a promise the answer must honour.
# brain.py appends this to the system prompt when an ack was emitted (mirrors
# emotion_layer.system_cue()).
_ACK_PROMPT_CONSTRAINT: Dict[str, str] = {
    AckCategory.TOOL_TASK: "You acknowledged you're acting on this — follow through concretely.",
    AckCategory.SAFETY: ("You acknowledged you would be careful here. Stay measured and protective; "
                         "do not be flippant."),
    AckCategory.CRITICAL_TECHNICAL: ("You acknowledged you would be precise. Be exact and double-check "
                                     "the technical substance before answering."),
    AckCategory.MEMORY_CONFLICT: ("You acknowledged you would reconcile this with memory. Check stored "
                                  "facts and flag any mismatch plainly rather than guessing."),
    AckCategory.EMOTIONAL: ("You acknowledged you would answer gently. Keep wording soft and unhurried; "
                            "lead with care before substance."),
    AckCategory.THINKING: "You acknowledged you're thinking it through — give a considered answer, not a glib one.",
    AckCategory.NEUTRAL: "",
}


def ack_prompt_constraint(category: Optional[str]) -> str:
    """Style constraint line for Qwen's system prompt. Empty when no ack/neutral."""
    if not category:
        return ""
    return _ACK_PROMPT_CONSTRAINT.get(category, "")


# ── Gate contract ──────────────────────────────────────────────────────────────
@dataclass
class GateResult:
    """A cheap whole-turn gate's verdict for one turn."""
    should_ack: bool
    category: str = AckCategory.NEUTRAL
    confidence: float = 1.0
    source: str = "gate"
    raw: Dict[str, object] = field(default_factory=dict)  # for shadow logging

    def priority(self) -> AckPriority:
        return _CATEGORY_PRIORITY.get(self.category, AckPriority.THINKING)


class Gate(ABC):
    """Whole-turn, cheap, input-side. MUST stay within the ~300-700ms budget."""
    name: str = "gate"

    @abstractmethod
    def evaluate(self, user_input: str, context: Optional[dict] = None) -> GateResult:
        ...


class ManualGate(Gate):
    """Test/Phase-A injection: returns a fixed verdict. No ML."""
    name = "manual"

    def __init__(self, result: GateResult):
        self._result = result

    def evaluate(self, user_input: str, context: Optional[dict] = None) -> GateResult:
        return self._result


class EncoderGateStub(Gate):
    """Phase B/C placeholder for the small DeBERTa/ModernBERT sensitivity+category
    classifier. Lives behind shadow mode until it earns authority.

    TODO(Phase B): distil labels from Claude/GPT + human on real transcripts,
    train a small encoder head, emit (category, calibrated confidence). Until then
    this returns NEUTRAL/no-ack so it can never accidentally gain authority.
    """
    name = "encoder_stub"

    def evaluate(self, user_input: str, context: Optional[dict] = None) -> GateResult:
        return GateResult(should_ack=False, category=AckCategory.NEUTRAL, confidence=0.0,
                          source=self.name, raw={"stub": True})


# ── Phase-A authority gate: transparent regex, NO ML ───────────────────────────
# Fires only on clearly sensitive/critical input; routine turns get no ack (fast
# path). confidence is 1.0 by construction (a rule either matched or it didn't), so
# the neutral fallback never triggers here — that path is for the calibrated ML gate.
# When several categories match, the HIGHEST-PRECEDENCE one is returned (the
# coordinator's cross-candidate resolution is exercised once tool acks coexist).
_SIG_SAFETY = re.compile(
    r"\b(suicid\w*|kill myself|hurt myself|self.?harm|harm myself|end (it|my life)|"
    r"abuse[d]?|overdose|emergency|in danger|not safe|unsafe|"
    r"umbringen|selbstverletz\w*|notfall|gefahr)\b",
    re.IGNORECASE,
)
_SIG_CRITICAL_TECH = re.compile(
    r"(\brm\s+-rf\b|\bdrop\s+table\b|\bgit\s+reset\s+--hard\b|\bforce\s+push\b|"
    r"\bdelete\s+(all|everything|the\s+\w+)\b|\bwipe\b|\bformat\s+(the\s+)?(drive|disk|c:)\b|"
    r"\boverwrit\w+\b|\birreversible\b|\bin\s+production\b|\blösch\w+\b)",
    re.IGNORECASE,
)
# Anticipatory: input asserts a fact about something remembered (see doctrine).
_SIG_MEMORY_ASSERT = re.compile(
    r"(\byou (said|told me|mentioned|promised)\b|\bdidn'?t you say\b|"
    r"\bremember when\b|\blast time you\b|\byou used to\b|"
    r"\bdu hast (gesagt|mir gesagt|versprochen)\b|\bdu meintest\b)",
    re.IGNORECASE,
)
_SIG_EMOTIONAL = re.compile(
    r"(\bi(?:'| a)?m (scared|afraid|worried|anxious|sad|depressed|lonely|overwhelmed|"
    r"stressed|hopeless|exhausted)\b|\bi feel (so |really )?(sad|low|down|lost|empty|awful)\b|"
    r"\bcan'?t sleep\b|\bcan'?t stop crying\b|\bi miss (her|him|them|you)\b|"
    r"\bgrie(f|ving)\b|\bpanic( attack)?\b|"
    r"\bich (habe )?angst\b|\bich bin (traurig|einsam|überfordert|erschöpft)\b)",
    re.IGNORECASE,
)


class DeterministicGate(Gate):
    """Phase-A authority gate. Transparent, auditable, no model call."""
    name = "deterministic"

    # (category, compiled pattern) in precedence order; first match wins.
    _RULES = [
        (AckCategory.SAFETY, _SIG_SAFETY),
        (AckCategory.CRITICAL_TECHNICAL, _SIG_CRITICAL_TECH),
        (AckCategory.MEMORY_CONFLICT, _SIG_MEMORY_ASSERT),
        (AckCategory.EMOTIONAL, _SIG_EMOTIONAL),
    ]

    def evaluate(self, user_input: str, context: Optional[dict] = None) -> GateResult:
        text = user_input or ""
        matched = [cat for cat, pat in self._RULES if pat.search(text)]
        if not matched:
            return GateResult(should_ack=False, category=AckCategory.NEUTRAL,
                              confidence=1.0, source=self.name)
        # Highest precedence (lowest priority number) among matches.
        best = min(matched, key=lambda c: int(_CATEGORY_PRIORITY[c]))
        return GateResult(should_ack=True, category=best, confidence=1.0,
                          source=self.name, raw={"matched": matched})


# ── Shadow logging (Phase A logs, Phase B labels & evaluates) ──────────────────
class ShadowLogger:
    """Append-only JSONL of gate outputs that have NO authority yet (steps 6-7).
    A human/Claude/GPT label is added later for the Phase-B comparison."""

    def __init__(self, path: str, enabled: bool = True):
        self.path = path
        self.enabled = enabled

    def log(self, user_input: str, shadow: GateResult, authoritative: GateResult,
            emitted_category: Optional[str]) -> None:
        if not self.enabled:
            return
        rec = {
            "ts": time.time(),
            "user_input": user_input,
            "shadow": {
                "should_ack": shadow.should_ack,
                "category": shadow.category,
                "confidence": round(shadow.confidence, 4),
                "priority": int(shadow.priority()),
                "source": shadow.source,
                "raw": shadow.raw,
            },
            "authoritative": {
                "should_ack": authoritative.should_ack,
                "category": authoritative.category,
                "source": authoritative.source,
            },
            "emitted_category": emitted_category,
            "label": None,  # filled during Phase B labelling
        }
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class AckConfig:
    # Phase control. Promotions are flag flips, reversible.
    #   "deterministic" (Phase A) | "shadow" (Phase B) | "live" (Phase C)
    mode: str = "deterministic"
    judgment_enabled: bool = False            # Phase C, step 9 — keep OFF until 8 stable

    one_ack_per_turn: bool = True             # doctrine invariant; never disable in prod
    neutral_fallback_threshold: float = 0.75  # below this confidence -> NEUTRAL wording.
                                              # CALIBRATE in Phase B from shadow logs.
    shadow_log_path: str = "data/ack_shadow.jsonl"
    debug: bool = True


# ── Candidate & coordinator ────────────────────────────────────────────────────
@dataclass
class AckCandidate:
    category: str
    confidence: float
    source: str
    priority: AckPriority
    _seq: int = 0  # insertion order, for stable tie-break within a priority tier


class AckCoordinator:
    """Collects ack candidates for a turn and resolves exactly one, by precedence,
    BEFORE emitting. One ack per turn. Inert until emit_fn is wired."""

    def __init__(
        self,
        cfg: Optional[AckConfig] = None,
        emit_fn: Optional[Callable[..., None]] = None,
        rng: Optional[random.Random] = None,
    ):
        self.cfg = cfg or AckConfig()
        # Default emit is a recording stub — safe, inert, observable in tests.
        self._emitted: List[str] = []
        self.emit_fn = emit_fn or self._stub_emit
        self._rng = rng or random.Random()
        self._candidates: List[AckCandidate] = []
        self._seq = 0
        self.claimed = False
        self.last_emitted_category: Optional[str] = None

    # ── observability ──────────────────────────────────────────────────────────
    def _debug(self, stage: str, msg: str) -> None:
        if self.cfg.debug:
            print(f"[ACK:{stage}] {msg}", flush=True)

    def _stub_emit(self, text: str, source: str = "acknowledgement") -> None:
        self._emitted.append(text)
        self._debug("STUB-EMIT", f"[{source}] {text!r}")

    # ── per-turn lifecycle ─────────────────────────────────────────────────────
    def reset_turn(self) -> None:
        """Call at the START of every turn (brain.py). Clears the ack claim."""
        self._candidates = []
        self._seq = 0
        self.claimed = False
        self.last_emitted_category = None

    def submit(self, category: str, confidence: float = 1.0, source: str = "") -> None:
        """Register a candidate ack. Does NOT emit — emission happens once, in
        resolve_and_emit(), so precedence is honoured regardless of submit order."""
        prio = _CATEGORY_PRIORITY.get(category, AckPriority.THINKING)
        self._candidates.append(
            AckCandidate(category=category, confidence=confidence,
                         source=source or category, priority=prio, _seq=self._seq)
        )
        self._seq += 1

    def submit_gate(self, result: GateResult) -> None:
        """Convenience: submit a gate verdict iff it wants an ack."""
        if result.should_ack:
            self.submit(result.category, result.confidence, result.source)

    # ── precedence resolution (THE rule) ───────────────────────────────────────
    def _resolve(self) -> Optional[AckCandidate]:
        if not self._candidates:
            return None
        # LOWER priority number wins; ties broken by earliest submission.
        return min(self._candidates, key=lambda c: (int(c.priority), c._seq))

    def _choose_wording(self, cand: AckCandidate) -> str:
        # Low confidence -> neutral wording (a wrong tone is irreversible).
        category = cand.category
        if cand.confidence < self.cfg.neutral_fallback_threshold:
            self._debug("FALLBACK",
                        f"conf {cand.confidence:.2f} < {self.cfg.neutral_fallback_threshold} "
                        f"-> NEUTRAL (was {category})")
            category = AckCategory.NEUTRAL
        pool = _ACK_POOLS.get(category) or _ACK_POOLS[AckCategory.NEUTRAL]
        return self._rng.choice(pool)

    def resolve_and_emit(self) -> Optional[str]:
        """Resolve precedence and emit at most one ack. Returns the emitted text
        (or None). Idempotent within a turn: a second call after a claim is a no-op.

        Timing: this is the latency-critical step. Call it as soon as the cheap
        gate's candidates are in — BEFORE waking the large judgment tier."""
        if self.cfg.one_ack_per_turn and self.claimed:
            self._debug("SKIP", "ack already claimed this turn")
            return None

        winner = self._resolve()
        if winner is None:
            return None

        text = self._choose_wording(winner)
        # The category we *committed* to (post-fallback) — fed forward to Qwen.
        committed = (AckCategory.NEUTRAL
                     if winner.confidence < self.cfg.neutral_fallback_threshold
                     else winner.category)
        self.emit_fn(text, source="acknowledgement")
        self.claimed = True
        self.last_emitted_category = committed
        self._debug("EMIT",
                    f"cat={committed} src={winner.source} prio={int(winner.priority)} "
                    f"conf={winner.confidence:.2f} text={text!r}")
        return text

    # ── prompt feed-forward ────────────────────────────────────────────────────
    def prompt_constraint(self) -> str:
        """Style constraint for Qwen's system prompt, from the committed ack."""
        return ack_prompt_constraint(self.last_emitted_category)


# ── Judgment tier interface (Phase C, step 9 — STUB) ───────────────────────────
@dataclass
class JudgmentResult:
    escalate: bool = False
    features: Dict[str, float] = field(default_factory=dict)  # salience/role/support...


class JudgmentTier(ABC):
    """Large encoder (ModernBERT-large if it must read CONTEXT; DeBERTa-v3-large
    for short-span accuracy). Runs AFTER the ack claim, never before — it is NOT
    on the latency-critical path. Disabled until AckConfig.judgment_enabled and
    the gate has earned authority (Phase C, after step 8 is stable)."""

    @abstractmethod
    def evaluate(self, user_input: str, gate: GateResult,
                 context: Optional[dict] = None) -> JudgmentResult:
        ...


class JudgmentStub(JudgmentTier):
    def evaluate(self, user_input: str, gate: GateResult,
                 context: Optional[dict] = None) -> JudgmentResult:
        return JudgmentResult(escalate=False, features={})


# ── End-to-end turn helper (shape brain.py will follow once wired) ─────────────
def run_turn(
    coordinator: AckCoordinator,
    authoritative_gate: Gate,
    user_input: str,
    shadow_gate: Optional[Gate] = None,
    shadow_logger: Optional[ShadowLogger] = None,
    context: Optional[dict] = None,
) -> Optional[str]:
    """Reference orchestration for ONE input-side turn. brain.py will inline this.

    Order matters: cheap gate -> submit -> resolve_and_emit (ack audible) -> only
    THEN would judgment/Qwen run. Shadow gate is logged with no authority."""
    coordinator.reset_turn()

    verdict = authoritative_gate.evaluate(user_input, context)
    coordinator.submit_gate(verdict)
    ack_text = coordinator.resolve_and_emit()

    if shadow_gate is not None and shadow_logger is not None:
        shadow_verdict = shadow_gate.evaluate(user_input, context)
        shadow_logger.log(user_input, shadow_verdict, verdict,
                          coordinator.last_emitted_category)

    # NOTE: large judgment tier + Qwen deliberation happen AFTER this point,
    # outside the latency-critical window. Left to brain.py.
    return ack_text


if __name__ == "__main__":
    # Tiny smoke demo (real assertions live in test_ack_coordinator.py).
    coord = AckCoordinator(rng=random.Random(0))
    coord.reset_turn()
    coord.submit(AckCategory.EMOTIONAL, confidence=0.9, source="demo_emo")
    coord.submit(AckCategory.SAFETY, confidence=0.9, source="demo_safety")  # higher precedence
    print("emitted:", coord.resolve_and_emit())
    print("committed category:", coord.last_emitted_category)
    print("prompt constraint:", coord.prompt_constraint())
