"""Calibration ceiling probe for the small long-NLI salience model.
Re-scores the same 15 candidates under varying transient_weight, and for each
finds the BEST single threshold + its accuracy. Tells us whether the swap is
recoverable by calibration or the model is fundamentally less separable here."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from salience_layer import SalienceLayer, SalienceConfig
from test_salience_compare import CANDIDATES

def best_threshold(scores_labels):
    # scores_labels: list of (score, gt_bool). Try every midpoint threshold.
    cand_th = sorted({round(s, 3) for s, _ in scores_labels})
    best = (0, 0.0)
    for th in cand_th:
        acc = sum((s >= th) == gt for s, gt in scores_labels)
        if acc > best[0]:
            best = (acc, th)
    return best

for tw in (0.30, 0.50, 0.70, 1.00):
    layer = SalienceLayer(SalienceConfig(debug=False, use_model=True,
                                         transient_weight=tw))
    sl = [(layer.score(t).score, gt) for t, gt, _ in CANDIDATES]
    acc, th = best_threshold(sl)
    # separability gap: min STORE score - max SKIP score (positive = clean split exists)
    store = [s for s, gt in sl if gt]
    skip  = [s for s, gt in sl if not gt]
    gap = min(store) - max(skip)
    print(f"transient_weight={tw:.2f}  best_acc={acc}/15 @ th={th:.2f}  "
          f"separability_gap={gap:+.2f}  (min_store={min(store):.2f} max_skip={max(skip):.2f})")
