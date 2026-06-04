# confidence_layer.py
"""
Vivianna V1 — Confidence Layer.

Implements the Cognitive Doctrine's confidence pattern:

    Event -> Judge -> Score -> Behavior Change

Doctrine (Confidence Doctrine):
  - Question: "How sure am I?"
  - Output: a confidence_score in [0, 1] — a FIRST-CLASS system variable.
  - High confidence   -> direct response.
  - Medium confidence -> softer wording / qualifiers.
  - Low confidence    -> flag uncertainty plainly.
  - Trust Doctrine: honest limitation over confident fabrication.

Design note:
  The router already gates *routing*-low confidence (< 0.70 NLI) into a
  clarification BEFORE generation. This layer governs WORDING DURING generation,
  using two signals that already exist in the system but were previously unused
  together:
    - nli_confidence:    routing certainty from router.nli_classify()
    - memory grounding:  confidence/validity from the stabilizer's pre_generate()
  Deterministic (doctrine: V1 = deterministic judges). No extra model call.

PURE module — imports nothing from the project. brain.py owns an instance and
merges the returned cue into the system prompt alongside the emotion cue.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ConfidenceResult:
    score: float
    band: str          # "high" | "medium" | "low"
    cue: str
    source: str = ""   # short trace of what drove the score


@dataclass
class ConfidenceConfig:
    debug: bool = True

    high: float = 0.75          # >= high  -> direct
    low: float = 0.45           # <  low   -> flag uncertainty
    default_nli: float = 0.80   # assumed routing certainty if none supplied

    terse_word_count: int = 2   # inputs this short get a small clarity penalty
    terse_penalty: float = 0.15
    vague_penalty: float = 0.10

    cues: Dict[str, str] = field(default_factory=lambda: {
        "high": "",
        "medium": ("Confidence is moderate. Use light qualifiers where appropriate "
                   "and avoid overclaiming certainty."),
        "low": ("Confidence is low. Be explicit about what you are unsure of rather "
                "than guessing — it is fine to say plainly what you don't know."),
    })


# Vague references that, with little context, lower certainty.
_VAGUE = re.compile(
    r"\b(this|that|it|those|these|the thing|the one|that one|das ding|"
    r"die sache|das da)\b",
    re.IGNORECASE,
)


class ConfidenceLayer:
    def __init__(self, cfg: ConfidenceConfig | None = None):
        self.cfg = cfg or ConfidenceConfig()
        self.last = ConfidenceResult(score=1.0, band="high", cue="", source="init")

    def _debug(self, msg: str) -> None:
        if self.cfg.debug:
            print(f"[CONFIDENCE] {msg}", flush=True)

    def assess(
        self,
        nli_confidence: float | None,
        memory_confidence: float,
        memory_valid: bool,
        user_input: str,
    ) -> ConfidenceResult:
        trace = []
        score = self.cfg.default_nli if nli_confidence is None else float(nli_confidence)
        trace.append(f"nli={score:.2f}")

        # Memory grounding: retrieved + valid lifts certainty; retrieved + invalid caps it.
        if memory_confidence > 0.0:
            if memory_valid:
                score = 0.6 * score + 0.4 * memory_confidence
                trace.append(f"mem+{memory_confidence:.2f}")
            else:
                score = min(score, 0.5)
                trace.append("mem_uncertain_cap")

        # Input clarity.
        text = (user_input or "").strip()
        if len(text.split()) <= self.cfg.terse_word_count:
            score -= self.cfg.terse_penalty
            trace.append(f"terse-{self.cfg.terse_penalty:.2f}")
        if _VAGUE.search(text):
            score -= self.cfg.vague_penalty
            trace.append(f"vague-{self.cfg.vague_penalty:.2f}")

        score = max(0.0, min(1.0, score))
        if score >= self.cfg.high:
            band = "high"
        elif score < self.cfg.low:
            band = "low"
        else:
            band = "medium"

        result = ConfidenceResult(
            score=round(score, 3),
            band=band,
            cue=self.cfg.cues.get(band, ""),
            source=" ".join(trace),
        )
        self.last = result
        self._debug(f"score={result.score:.2f} band={band} [{result.source}]")
        return result
