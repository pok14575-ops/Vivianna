# salience_layer.py
"""
Vivianna V1 — Salience Layer.

Implements the Cognitive Doctrine's universal pattern for salience:

    Event -> Judge -> Score -> (State / Behavior)

Doctrine (Salience Doctrine + Memory Doctrine):
  - Question: "Does this matter?"
  - Output: a salience_score in [0, 1].
  - Prefer: user identity, lasting preferences, long-term projects,
    emotional significance, recurring themes — over transient details.
  - "Remember what matters", not "remember everything".
  - Uses: storage gating, retrieval ranking, future compression.

PRIMARY judge: a DeBERTa zero-shot semantic scorer (when a model is configured
and loads successfully). FALLBACK judge: the deterministic regex scorer below,
used when the model is disabled or unavailable. The regex path imports nothing
from the project and nothing heavy, so the module stays unit-testable in
isolation (the model is lazy-imported only when first used).

brain.py decides how the score is used (store / skip); memory.py uses the
stored score for optional retrieval re-ranking. The public contract is
unchanged: SalienceLayer.score(text, recurring) -> SalienceResult(score, reasons).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Pattern


@dataclass
class SalienceResult:
    score: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class SalienceConfig:
    debug: bool = True

    base: float = 0.10
    recurring_boost: float = 0.20
    short_text_chars: int = 15        # below this, treat as low-value fragment
    short_text_penalty: float = 0.20

    # --- Semantic model (PRIMARY when available) ---
    use_model: bool = True
    model_name: str = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
    model_device: str = "auto"        # "auto" -> cuda if available else cpu; or "cuda"/"cpu"
    transient_weight: float = 0.30    # penalty multiplier on P(transient) for the model path

    # Natural-language hypotheses for the zero-shot model. Keys are the facet
    # names; "transient" is the negative facet (subtracts from the score).
    hypotheses: Dict[str, str] = field(default_factory=lambda: {
        "identity":   "This states who the user or their family is — their name, job, background, or relationships.",
        "preference": "This states a lasting preference, like, dislike, habit, or value of the user.",
        "project":    "This is about the user's long-term project, goal, or ongoing work.",
        "emotional":  "This expresses something emotionally significant or a major life event for the user.",
        "health":     "This states a health condition, allergy, dietary restriction, or other lasting personal constraint.",
        "transient":  "This is a transient, time-bound, or throwaway detail that is not worth remembering.",
    })

    # Category weights for the REGEX fallback. A memory matching a category gets
    # its weight added once.
    weights: Dict[str, float] = field(default_factory=lambda: {
        "identity": 0.45,     # who the user / family is
        "preference": 0.35,   # lasting likes/dislikes/habits
        "project": 0.35,      # long-term work, goals, Vivianna itself
        "emotional": 0.25,    # emotional significance, relationships
        "transient": -0.30,   # time-bound / throwaway -> pushes score down
    })

    patterns: Dict[str, Pattern] = field(default_factory=lambda: {
        "identity": re.compile(
            r"\b(my name is|i am (a|an|the)\b|i'?m (a|an|the)\b|call me\b|"
            r"i live in|i work (as|at)|my (wife|husband|daughter|son|kid|child|"
            r"children|family|mother|father|mom|dad|brother|sister|partner|name|job|"
            r"birthday|address)|ich heiße|ich bin (ein|eine|der|die)|"
            r"meine? (frau|mann|tochter|sohn|kind|familie|name))\b",
            re.IGNORECASE,
        ),
        "preference": re.compile(
            r"\b(i (like|love|hate|prefer|enjoy|dislike|can'?t stand|always|never)\b|"
            r"my favou?rite|i'?m allergic|i don'?t (eat|drink|like)|"
            r"ich (mag|liebe|hasse|bevorzuge)|mein lieblings|ich bin allergisch)\b",
            re.IGNORECASE,
        ),
        "project": re.compile(
            r"\b(project|working on|i'?m building|we'?re building|my goal|i plan to|"
            r"trying to build|long[- ]term|vivianna|the assistant|the avatar|"
            r"projekt|ich arbeite an|mein ziel|ich baue|wir bauen)\b",
            re.IGNORECASE,
        ),
        "emotional": re.compile(
            r"\b(important to me|means a lot|i miss|i'?m worried|i'?m scared|"
            r"i'?m proud|i'?m grateful|i care about|wichtig für mich|ich vermisse|"
            r"ich mache mir sorgen|ich bin stolz)\b",
            re.IGNORECASE,
        ),
        "transient": re.compile(
            r"\b(today|right now|currently|this morning|this afternoon|tonight|"
            r"just now|at the moment|the weather|what time|temperature|"
            r"heute|gerade|im moment|das wetter|wie spät)\b",
            re.IGNORECASE,
        ),
    })


class SalienceLayer:
    def __init__(self, cfg: SalienceConfig | None = None):
        self.cfg = cfg or SalienceConfig()
        self._clf = None              # lazy zero-shot pipeline
        self._model_failed = False    # set True once load fails -> stay on regex
        self._model_labels: List[str] = []  # hypothesis strings, model order
        self._label2facet: Dict[str, str] = {}

    def _debug(self, msg: str) -> None:
        if self.cfg.debug:
            print(f"[SALIENCE] {msg}", flush=True)

    # ── model loading (lazy, fallback-safe) ──────────────────────────────────
    def _ensure_model(self) -> bool:
        """Load the zero-shot model on first use. Returns True if usable."""
        if not self.cfg.use_model or self._model_failed:
            return False
        if self._clf is not None:
            return True
        try:
            import torch  # noqa: WPS433 (lazy, optional dependency)
            from transformers import pipeline

            if self.cfg.model_device == "auto":
                device = 0 if torch.cuda.is_available() else -1
            elif self.cfg.model_device == "cuda":
                device = 0
            else:
                device = -1

            self._clf = pipeline(
                "zero-shot-classification",
                model=self.cfg.model_name,
                device=device,
                dtype=torch.float32,
            )
            self._label2facet = {v: k for k, v in self.cfg.hypotheses.items()}
            self._model_labels = list(self.cfg.hypotheses.values())
            self._debug(
                f"model loaded: {self.cfg.model_name} "
                f"device={'cuda' if device == 0 else 'cpu'}"
            )
            return True
        except Exception as e:  # noqa: BLE001 — any failure -> regex fallback
            self._model_failed = True
            self._clf = None
            self._debug(f"model load FAILED ({type(e).__name__}: {e}); using regex fallback")
            return False

    # ── scorers ──────────────────────────────────────────────────────────────
    def _score_model(self, t: str, recurring: bool) -> Optional[SalienceResult]:
        """Semantic salience via zero-shot. Returns None on any runtime failure."""
        try:
            out = self._clf(t, self._model_labels, multi_label=True)
            probs = {self._label2facet[lbl]: float(p)
                     for lbl, p in zip(out["labels"], out["scores"])}
        except Exception as e:  # noqa: BLE001
            self._debug(f"model inference failed ({type(e).__name__}: {e}); regex fallback")
            return None

        memorable_facets = {k: v for k, v in probs.items() if k != "transient"}
        top_facet = max(memorable_facets, key=memorable_facets.get)
        memorable = memorable_facets[top_facet]
        p_transient = probs.get("transient", 0.0)

        s = memorable - self.cfg.transient_weight * p_transient
        reasons = [
            "model",
            f"{top_facet}={memorable:.2f}",
            f"transient-{self.cfg.transient_weight * p_transient:.2f}",
        ]

        if len(t) < self.cfg.short_text_chars:
            s -= self.cfg.short_text_penalty
            reasons.append(f"short-{self.cfg.short_text_penalty:.2f}")
        if recurring:
            s += self.cfg.recurring_boost
            reasons.append(f"recurring+{self.cfg.recurring_boost:.2f}")

        s = max(0.0, min(1.0, s))
        self._debug(f"score={s:.2f} reasons={reasons} text={t[:60]!r}")
        return SalienceResult(score=round(s, 3), reasons=reasons)

    def _score_regex(self, t: str, recurring: bool) -> SalienceResult:
        """Deterministic regex judge (fallback). Original V1 logic."""
        reasons: List[str] = []
        s = self.cfg.base

        for name, pat in self.cfg.patterns.items():
            if pat.search(t):
                w = self.cfg.weights.get(name, 0.0)
                s += w
                reasons.append(f"{name}{'+' if w >= 0 else ''}{w:.2f}")

        if len(t) < self.cfg.short_text_chars:
            s -= self.cfg.short_text_penalty
            reasons.append(f"short-{self.cfg.short_text_penalty:.2f}")
        if recurring:
            s += self.cfg.recurring_boost
            reasons.append(f"recurring+{self.cfg.recurring_boost:.2f}")

        s = max(0.0, min(1.0, s))
        reasons.insert(0, "regex")
        self._debug(f"score={s:.2f} reasons={reasons} text={t[:60]!r}")
        return SalienceResult(score=round(s, 3), reasons=reasons)

    # ── public API (unchanged contract) ──────────────────────────────────────
    def score(self, text: str, recurring: bool = False) -> SalienceResult:
        """Judge how much a candidate memory matters. [0, 1].

        Uses the semantic model when available; falls back to the regex judge
        when the model is disabled, fails to load, or errors at inference.
        """
        t = (text or "").strip()

        if self._ensure_model():
            res = self._score_model(t, recurring)
            if res is not None:
                return res
        return self._score_regex(t, recurring)
