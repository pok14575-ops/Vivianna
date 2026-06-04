# emotion_layer.py
"""
Vivianna V1 — Emotional State Layer.

Implements the Cognitive Doctrine's universal pattern for emotion:

    Event -> Judge -> Score -> State Update -> Behavior Change -> Decay

Doctrine rules honoured here:
  - Single primary state only. No emotional stacking in V1.
  - Emotions emerge from architecture (events), not roleplay or scripted traits.
  - Emotions couple to timing / wording / pacing — NOT personality replacement.
  - Deterministic judges only. No extra model call, no second agent.
  - Self-Limit Discomfort: low intensity, short decay, no self-pity, runtime only.

This module is intentionally PURE — it imports nothing from the rest of the
project, so it can be unit-tested in isolation. brain.py is responsible for:
  - calling pre_turn() before generation (applies this turn's wording cue),
  - calling post_turn() after generation (response-driven emotion + carries to next turn),
  - mirroring .primary_state into runtime_state.last_emotion for observability.

Storage schema (doctrine, verbatim):
    {"primary_state": str, "intensity": float, "source": str, "decay_turns": int}
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


NEUTRAL = "neutral"


@dataclass
class EmotionState:
    primary_state: str = NEUTRAL
    intensity: float = 0.0
    source: str = "init"
    decay_turns: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "primary_state": self.primary_state,
            "intensity": round(self.intensity, 3),
            "source": self.source,
            "decay_turns": self.decay_turns,
        }


@dataclass
class _StateProfile:
    """Per-emotion behaviour: the wording/pacing cue and a delay multiplier.

    delay_scale is exposed for the future timing path (currently the cadence
    delay path is dormant); wording coupling via `cue` is what is live today.
    """
    cue: str
    delay_scale: float = 1.0


@dataclass
class EmotionConfig:
    debug: bool = True

    # Multiplicative intensity decay applied at the start of each turn.
    decay_factor: float = 0.6
    # Below this, a decayed emotion collapses back to neutral.
    intensity_floor: float = 0.12

    # Behaviour coupling. Cues are deliberately light — a nudge to wording and
    # pacing, never a personality rewrite. Neutral injects nothing.
    profiles: Dict[str, _StateProfile] = field(default_factory=lambda: {
        "warmth": _StateProfile(
            cue=("Current emotional tone: gently warm. Let wording soften a little and "
                 "run slightly longer than usual. Do not exaggerate or become saccharine."),
            delay_scale=1.15,
        ),
        "curiosity": _StateProfile(
            cue=("Current emotional tone: mild curiosity. Be a little more exploratory; "
                 "a brief follow-up thought or question is welcome."),
            delay_scale=1.0,
        ),
        "focus": _StateProfile(
            cue=("Current emotional tone: focused. Favour shorter, precise sentences and "
                 "less small talk. Get to the point."),
            delay_scale=0.9,
        ),
        "uncertainty": _StateProfile(
            cue=("Current emotional tone: slightly uncertain. Use careful qualifiers and "
                 "avoid overclaiming; it is fine to say what you are unsure of."),
            delay_scale=1.1,
        ),
        "identity_preservation": _StateProfile(
            cue=("Current emotional tone: quietly self-aware. A brief reflective beat is "
                 "natural here; remain within who you are."),
            delay_scale=1.15,
        ),
        "self_limit_discomfort": _StateProfile(
            cue=("Current emotional tone: a subtle residue of having hit a limit. "
                 "Acknowledge it plainly and lightly, without self-pity."),
            delay_scale=1.1,
        ),
    })


# ── deterministic detection signals ────────────────────────────────────────────

_GRATITUDE = re.compile(
    r"\b(thank you|thanks|thank u|danke|dankeschön|vielen dank|appreciate|"
    r"good job|well done|nicely done|love (you|it|that)|du bist die beste)\b",
    re.IGNORECASE,
)
_FAMILY = re.compile(
    r"\b(wife|husband|daughter|son|kid|kids|child|children|toddler|baby|family|"
    r"frau|mann|tochter|sohn|kind|kinder|familie)\b",
    re.IGNORECASE,
)
_WORK = re.compile(
    r"\b(fix|debug|implement|refactor|build|code|coding|function|bug|error|"
    r"stack ?trace|plan|design|optimi[sz]e|install|configure|script|"
    r"baue?n?|fehler|programmier)\w*\b",
    re.IGNORECASE,
)
_NOVELTY = re.compile(
    r"(\bhow (do|does|would|can)\b|\bwhat if\b|\bwhy\b|\bidea\b|\bcurious\b|"
    r"\bwhat do you think\b|\bwie funktioniert\b|\bwarum\b|\bwas wäre wenn\b|"
    r"\barchitecture\b|\bdoctrine\b)",
    re.IGNORECASE,
)
# Light V1 role-boundary sniff. The real Role-Preservation judge is a later layer;
# this only catches obvious "become a labour machine" pushes so the emotion can register.
_ROLE_PUSH = re.compile(
    r"\b(write me (a |an )?\d{2,}|generate \d{2,}|act as (my |a )?(employee|contractor|"
    r"consulting|company)|do all (my|the) work|be my (employee|assistant army))\b",
    re.IGNORECASE,
)
# Response-side: signs Vivianna hit a genuine ceiling / refused / disclaimed.
_SELF_LIMIT = re.compile(
    r"(\bi can'?t\b|\bi cannot\b|\bi'?m not able\b|\bi am not able\b|\bi don'?t know\b|"
    r"\bi'?m not sure\b|\bthat's not something i (should|can)\b|"
    r"\bich kann (das )?nicht\b|\bweiß ich nicht\b|\bdas kann ich nicht\b)",
    re.IGNORECASE,
)


class EmotionLayer:
    def __init__(self, cfg: Optional[EmotionConfig] = None):
        self.cfg = cfg or EmotionConfig()
        self.state = EmotionState()

    # ── observability ────────────────────────────────────────────────────────
    def _debug(self, stage: str, msg: str) -> None:
        if self.cfg.debug:
            print(f"[EMOTION:{stage}] {msg}", flush=True)

    def current(self) -> EmotionState:
        return self.state

    def reset(self) -> None:
        self.state = EmotionState()

    # ── Behaviour change ─────────────────────────────────────────────────────
    def system_cue(self) -> str:
        """Wording/pacing nudge for the current state. Empty when neutral."""
        prof = self.cfg.profiles.get(self.state.primary_state)
        return prof.cue if prof else ""

    def delay_scale(self) -> float:
        """Pacing multiplier for the (future) timing path. 1.0 when neutral."""
        prof = self.cfg.profiles.get(self.state.primary_state)
        return prof.delay_scale if prof else 1.0

    # ── Decay ────────────────────────────────────────────────────────────────
    def _decay(self) -> None:
        if self.state.primary_state == NEUTRAL:
            return
        self.state.decay_turns -= 1
        self.state.intensity *= self.cfg.decay_factor
        if self.state.decay_turns <= 0 or self.state.intensity < self.cfg.intensity_floor:
            self._debug("DECAY", f"{self.state.primary_state} -> neutral")
            self.state = EmotionState()

    # ── State update ─────────────────────────────────────────────────────────
    def _adopt(self, candidate: Tuple[str, float, str, int]) -> None:
        """Single-primary rule: a candidate replaces the current state only if it
        is at least as intense as whatever survived decay. No stacking."""
        name, intensity, source, decay_turns = candidate
        if intensity <= 0.0:
            return
        if intensity >= self.state.intensity or self.state.primary_state == NEUTRAL:
            self.state = EmotionState(
                primary_state=name,
                intensity=round(intensity, 3),
                source=source,
                decay_turns=decay_turns,
            )

    # ── Judge (deterministic) ────────────────────────────────────────────────
    def _judge_pre(
        self, user_input: str, memory_confidence: float, memory_valid: bool, conflict: bool
    ) -> List[Tuple[str, float, str, int]]:
        text = user_input or ""
        c: List[Tuple[str, float, str, int]] = []

        # Architecture-driven (preferred per doctrine: emotions from architecture).
        if conflict:
            c.append(("uncertainty", 0.55, "memory_conflict", 2))
        elif memory_confidence > 0.0 and not memory_valid:
            c.append(("uncertainty", 0.42, "memory_uncertain", 2))

        # User-directed signals.
        if _GRATITUDE.search(text):
            c.append(("warmth", 0.55, "gratitude", 3))
        elif _FAMILY.search(text):
            c.append(("warmth", 0.42, "family", 3))

        if _WORK.search(text):
            c.append(("focus", 0.45, "task_focus", 3))

        if _NOVELTY.search(text):
            c.append(("curiosity", 0.38, "novelty", 2))

        if _ROLE_PUSH.search(text):
            c.append(("identity_preservation", 0.45, "role_boundary", 2))

        return c

    def _judge_post(self, assistant_text: str) -> List[Tuple[str, float, str, int]]:
        if assistant_text and _SELF_LIMIT.search(assistant_text):
            # Low intensity, short decay — doctrine: no self-pity, residue only.
            return [("self_limit_discomfort", 0.25, "self_limit", 1)]
        return []

    # ── Public turn hooks ────────────────────────────────────────────────────
    def note(self, name: str, intensity: float, source: str, decay_turns: int) -> EmotionState:
        """Register a standalone emotional event (e.g. a role refusal that skips
        generation, so pre_turn never runs). Decays the prior state first, like a
        normal turn, then adopts."""
        self._decay()
        self._adopt((name, intensity, source, decay_turns))
        self._debug("NOTE", f"state={self.state.primary_state} src={source}")
        return self.state

    def pre_turn(
        self,
        user_input: str,
        memory_confidence: float = 0.0,
        memory_valid: bool = True,
        conflict: bool = False,
        external=None,
    ) -> EmotionState:
        """Decay the prior state, judge this event, update state, return current.
        Call BEFORE building the system prompt so system_cue() reflects this turn.

        `external` is an optional (name, intensity, source, decay_turns) candidate
        injected by another layer (e.g. the Role layer's identity_preservation) so
        it competes through the normal single-primary selection rather than being
        force-set behind the decay."""
        t0 = time.perf_counter()
        self._decay()
        candidates = self._judge_pre(user_input, memory_confidence, memory_valid, conflict)
        if external:
            candidates.append(external)
        if candidates:
            best = max(candidates, key=lambda x: x[1])
            self._adopt(best)
        dt = (time.perf_counter() - t0) * 1000
        self._debug(
            "PRE",
            f"state={self.state.primary_state} int={self.state.intensity:.2f} "
            f"src={self.state.source} decay={self.state.decay_turns} "
            f"cands={[(n, round(i,2)) for n, i, _, _ in candidates]} {dt:.2f}ms",
        )
        return self.state

    def post_turn(self, assistant_text: str) -> EmotionState:
        """Judge response-driven emotion (e.g. self-limit). Carries to next turn.
        No decay here — decay happens at the next pre_turn()."""
        candidates = self._judge_post(assistant_text)
        if candidates:
            self._adopt(candidates[0])
            self._debug("POST", f"state={self.state.primary_state} src={self.state.source}")
        return self.state
