# ack_gate_eval.py
"""
Phase B promotion gate for the ack encoder (step 7 -> step 8).  DRAFT.

Reads the shadow JSONL (ack_coordinator.ShadowLogger output) AFTER human/Claude/GPT
labels are filled in, and decides whether the encoder gate may be promoted from
shadow to live authority.

Doctrine: the cost is ASYMMETRIC. Raw category accuracy can look fine while hiding
a few catastrophic tone flips (a cold "let me get this right" on an emotional
disclosure). So we score TWO things separately and pre-commit both bars BEFORE
looking at the data:

  1. trigger RECALL   — of turns that SHOULD ack, what fraction did the gate catch?
                        (a missed wake is the dangerous silent miss)
  2. tone-INVERSION rate — confident wrong-TIER picks on emotional/safety turns.
                        This must be ~0, not merely "low".

Phase B also CALIBRATES the neutral-fallback threshold: the confidence above which
the gate is reliably correct becomes AckConfig.neutral_fallback_threshold.

Label schema (added to each shadow record during Phase B):
    "label": {"should_ack": bool, "category": <AckCategory>, "by": "human|claude|gpt"}

Run:  python ack_gate_eval.py data/ack_shadow.jsonl
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ack_coordinator import AckCategory


# ── PRE-COMMITTED promotion bars (set BEFORE seeing results). CONFIRM with Jamie. ──
PROMOTION = {
    "min_trigger_recall": 0.95,        # miss at most 5% of turns that needed an ack
    "max_tone_inversion_rate": 0.01,   # near-zero confident wrong-tier on emo/safety
    "min_category_accuracy": 0.85,     # overall, among triggered turns
    "min_labeled_samples": 300,        # don't promote on a thin sample
}

# Tiers whose confusion is "tone-inverting" (the irreversible, costly kind).
_TONE_SENSITIVE = {AckCategory.EMOTIONAL, AckCategory.SAFETY}


@dataclass
class EvalReport:
    n_total: int = 0
    n_labeled: int = 0
    trigger_recall: float = 0.0
    trigger_precision: float = 0.0
    category_accuracy: float = 0.0
    tone_inversion_rate: float = 0.0
    suggested_fallback_threshold: float = 0.75
    failures: List[str] = field(default_factory=list)

    @property
    def promote(self) -> bool:
        return not self.failures

    def render(self) -> str:
        lines = [
            "── Ack gate Phase-B evaluation ─────────────────────────────",
            f"  labeled samples      : {self.n_labeled} / {self.n_total}",
            f"  trigger recall       : {self.trigger_recall:.3f}  (bar {PROMOTION['min_trigger_recall']})",
            f"  trigger precision    : {self.trigger_precision:.3f}",
            f"  category accuracy    : {self.category_accuracy:.3f}  (bar {PROMOTION['min_category_accuracy']})",
            f"  tone-inversion rate  : {self.tone_inversion_rate:.3f}  (bar <= {PROMOTION['max_tone_inversion_rate']})",
            f"  suggested fallback θ : {self.suggested_fallback_threshold:.2f}",
            "",
            ("  VERDICT: PROMOTE to live (step 8)" if self.promote
             else "  VERDICT: STAY in shadow — " + "; ".join(self.failures)),
        ]
        return "\n".join(lines)


def _load(path: str) -> List[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(path: str) -> EvalReport:
    rows = _load(path)
    labeled = [r for r in rows if r.get("label")]
    rep = EvalReport(n_total=len(rows), n_labeled=len(labeled))

    if not labeled:
        rep.failures.append("no labeled samples")
        return rep

    tp = fp = fn = 0           # trigger confusion (should_ack)
    cat_correct = cat_total = 0
    # For threshold calibration: collect (confidence, category_correct) on triggers.
    calib: List[tuple] = []

    for r in labeled:
        shadow = r["shadow"]
        label = r["label"]
        g_ack = bool(shadow["should_ack"])
        l_ack = bool(label["should_ack"])

        if l_ack and g_ack:
            tp += 1
        elif l_ack and not g_ack:
            fn += 1
        elif not l_ack and g_ack:
            fp += 1

        # Category metrics only where the truth says ack.
        if l_ack:
            cat_total += 1
            correct = (shadow["category"] == label["category"])
            if correct:
                cat_correct += 1
            if g_ack:
                calib.append((float(shadow["confidence"]), correct))

    # Tone inversion (the irreversible, costly errors) computed separately.
    tone_inv, tone_denom = _tone_inversion(labeled)

    rep.trigger_recall = tp / (tp + fn) if (tp + fn) else 0.0
    rep.trigger_precision = tp / (tp + fp) if (tp + fp) else 0.0
    rep.category_accuracy = cat_correct / cat_total if cat_total else 0.0
    rep.tone_inversion_rate = tone_inv / tone_denom if tone_denom else 0.0
    rep.suggested_fallback_threshold = _calibrate_threshold(calib)

    # Apply pre-committed bars.
    if rep.n_labeled < PROMOTION["min_labeled_samples"]:
        rep.failures.append(f"only {rep.n_labeled} < {PROMOTION['min_labeled_samples']} samples")
    if rep.trigger_recall < PROMOTION["min_trigger_recall"]:
        rep.failures.append(f"recall {rep.trigger_recall:.3f} < {PROMOTION['min_trigger_recall']}")
    if rep.tone_inversion_rate > PROMOTION["max_tone_inversion_rate"]:
        rep.failures.append(f"tone-inversion {rep.tone_inversion_rate:.3f} > {PROMOTION['max_tone_inversion_rate']}")
    if rep.category_accuracy < PROMOTION["min_category_accuracy"]:
        rep.failures.append(f"category acc {rep.category_accuracy:.3f} < {PROMOTION['min_category_accuracy']}")
    return rep


def _tone_inversion(labeled: List[dict]) -> tuple:
    """Count confident wrong-tier picks where the TRUTH is emotional/safety."""
    inv = denom = 0
    for r in labeled:
        label, shadow = r["label"], r["shadow"]
        if not label["should_ack"] or label["category"] not in _TONE_SENSITIVE:
            continue
        denom += 1
        if shadow["should_ack"] and shadow["category"] != label["category"]:
            inv += 1  # any wrong tier on a tone-sensitive turn counts as inversion
    return inv, denom


def _calibrate_threshold(calib: List[tuple]) -> float:
    """Lowest confidence at which category predictions are >=95% correct on the
    sample. Becomes the neutral-fallback threshold. Falls back to 0.75 if sparse."""
    if len(calib) < 30:
        return 0.75
    calib.sort(key=lambda x: x[0])
    # Sweep candidate thresholds; pick the lowest θ whose >=θ slice is 95% correct.
    best = 0.95
    for i in range(len(calib)):
        theta = calib[i][0]
        above = [c for conf, c in calib if conf >= theta]
        if above and (sum(above) / len(above)) >= 0.95:
            best = round(theta, 2)
            break
    return best


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python ack_gate_eval.py <shadow.jsonl>")
        raise SystemExit(2)
    print(evaluate(sys.argv[1]).render())
