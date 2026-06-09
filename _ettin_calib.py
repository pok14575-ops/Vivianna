"""Scratch: calibrate ettin-150m as a salience scorer on the large set.

Diagnoses WHY ettin mis-scores (per-facet dump on the miss items), then sweeps
calibration knobs on the cached facet logits:
  - transient penalty weight w   (the 0.30 default was tuned for DeBERTa)
  - logit temperature T          (sharpen/squash sigmoid)
  - aggregation mode:
      indep   : sigmoid per facet, score = max(memorable) - w*P(transient)   [current]
      softmax : softmax over all facets, score = 1 - P(transient)            [facets compete]
Picks by AUC (threshold-free). Compares DeBERTa vs ettin-default vs ettin-best,
and includes an even/odd split sanity check on the winner (in-sample caveat).
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import test_reranker_compare as T
import reranker_eval_data_large as L
from salience_layer import SalienceConfig, SalienceLayer

CANDS = L.BIG_SAL_CANDIDATES
labels = [gt for _, gt in CANDS]
cfg = SalienceConfig()
facets = list(cfg.hypotheses.keys())          # identity..transient (transient last)
hyps = [cfg.hypotheses[f] for f in facets]
ti = facets.index("transient")
mem_idx = [j for j in range(len(facets)) if j != ti]
SHORT_CHARS, SHORT_PEN = cfg.short_text_chars, cfg.short_text_penalty

model = T.load_ce("cross-encoder/ettin-reranker-150m-v1", "cuda")
model.predict([["w", "w"]], show_progress_bar=False)

# cache raw facet logits once: [n_items, n_facets]
raw = np.zeros((len(CANDS), len(facets)))
for i, (text, _) in enumerate(CANDS):
    raw[i] = model.predict([[hyps[j], text] for j in range(len(facets))],
                           show_progress_bar=False)


def sig(x, temp=1.0):
    return 1.0 / (1.0 + np.exp(-x / temp))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def score_indep(i, w, temp):
    p = sig(raw[i], temp)
    s = p[mem_idx].max() - w * p[ti]
    if len(CANDS[i][0]) < SHORT_CHARS:
        s -= SHORT_PEN
    return max(0.0, min(1.0, s))


def score_softmax(i, temp):
    p = softmax(raw[i] / temp)
    s = 1.0 - p[ti]
    if len(CANDS[i][0]) < SHORT_CHARS:
        s -= SHORT_PEN
    return max(0.0, min(1.0, s))


def build(mode, w=0.3, temp=1.0):
    if mode == "indep":
        return [score_indep(i, w, temp) for i in range(len(CANDS))]
    return [score_softmax(i, temp) for i in range(len(CANDS))]


# ── diagnostic: per-facet sigmoid on the items ettin got wrong by default ────
MISS = [
    "afraid of losing his work", "built his own PC", "task right now is fixing",
    "greeted Vivianna good morning", "speaks German and Chinese", "birthday is in October",
    "feeling motivated right now",
]
print("PER-FACET sigmoid(logit) on the default-config miss items "
      "(facets: " + " ".join(facets) + "):")
for i, (text, gt) in enumerate(CANDS):
    if any(m in text for m in MISS):
        p = sig(raw[i], 1.0)
        cells = " ".join(f"{f[:4]}={p[j]:.2f}" for j, f in enumerate(facets))
        print(f"  [{'STORE' if gt else 'skip ':>5}] {cells}  :: {text[:48]}")


# ── sweep ────────────────────────────────────────────────────────────────────
configs = []
for w in [0.3, 0.5, 0.75, 1.0, 1.5, 2.0]:
    configs.append((f"indep   w={w:<4} T=1.0", build("indep", w, 1.0)))
for temp in [0.5, 1.5, 2.0]:
    configs.append((f"indep   w=1.0  T={temp}", build("indep", 1.0, temp)))
for temp in [0.5, 1.0, 2.0]:
    configs.append((f"softmax        T={temp}", build("softmax", temp=temp)))

print(f"\n{'config':<22}{'AUC':>6}{'best':>7}{'@t':>6}")
print("-" * 41)
results = []
for name, sc in configs:
    auc = T.manual_auc(sc, labels)
    bacc, bt = T.best_threshold(sc, labels)
    results.append((name, sc, auc, bacc, bt))
    print(f"{name:<22}{auc:>6.3f}{bacc*64:>5.0f}/64{bt:>6.2f}")

# ── compare: DeBERTa vs ettin-default vs ettin-best (by AUC) ──────────────────
layer = SalienceLayer(SalienceConfig(debug=False, use_model=True))
layer.score("warmup")
deb = [layer.score(t).score for t, _ in CANDS]
deb_auc = T.manual_auc(deb, labels)
deb_bacc, deb_bt = T.best_threshold(deb, labels)

default_sc = build("indep", 0.30, 1.0)
def_auc = T.manual_auc(default_sc, labels)
def_bacc, _ = T.best_threshold(default_sc, labels)

best = max(results, key=lambda r: (r[2], r[3]))  # by AUC then acc
print("\n" + "=" * 50)
print("COMPARISON (64-item adversarial salience)")
print("=" * 50)
print(f"{'model/config':<30}{'AUC':>6}{'best':>8}")
print(f"{'DeBERTa-large (incumbent)':<30}{deb_auc:>6.3f}{deb_bacc*64:>5.0f}/64")
print(f"{'ettin-150m default(w=0.3)':<30}{def_auc:>6.3f}{def_bacc*64:>5.0f}/64")
print(f"{'ettin-150m CALIBRATED ['+best[0].strip()+']':<30}{best[2]:>6.3f}{best[3]*64:>5.0f}/64")

# even/odd split sanity check on the winner vs default (in-sample overfit guard)
ev = list(range(0, len(CANDS), 2))
od = list(range(1, len(CANDS), 2))
def auc_sub(sc, idx):
    return T.manual_auc([sc[i] for i in idx], [labels[i] for i in idx])
print(f"\nsplit-AUC check (winner): even={auc_sub(best[1], ev):.3f} odd={auc_sub(best[1], od):.3f}"
      f"  | default even={auc_sub(default_sc, ev):.3f} odd={auc_sub(default_sc, od):.3f}")
