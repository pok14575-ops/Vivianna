"""Scratch: per-item salience miss diff, DeBERTa-large vs ettin-150m, on the
large adversarial set. Shows each model's decision at its OWN best threshold and
lists exactly which items each misses (and which they BOTH miss = ambiguous)."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import test_reranker_compare as T
import reranker_eval_data_large as L
from salience_layer import SalienceLayer, SalienceConfig

T.SAL_CANDIDATES = L.BIG_SAL_CANDIDATES          # so T.salience_scores_crossencoder uses it
CANDS = L.BIG_SAL_CANDIDATES
labels = [gt for _, gt in CANDS]

# DeBERTa incumbent
layer = SalienceLayer(SalienceConfig(debug=False, use_model=True))
layer.score("warmup")
deb = [layer.score(t).score for t, _ in CANDS]

# ettin-150m as facet scorer
model = T.load_ce("cross-encoder/ettin-reranker-150m-v1", "cuda")
model.predict([["w", "w"]], show_progress_bar=False)
ett = T.salience_scores_crossencoder(model)

_, dt = T.best_threshold(deb, labels)
_, et = T.best_threshold(ett, labels)
print(f"best thresholds: DeBERTa={dt:.3f}  ettin-150m={et:.3f}\n")

dd = [s >= dt for s in deb]
ed = [s >= et for s in ett]

print(f"{'gold':>5} {'deb':>5}{'d?':>3} {'ett':>5}{'e?':>3}  text")
print("-" * 92)
deb_miss, ett_miss, both_miss = [], [], []
for (text, gt), ds, dok, es, eok in zip(CANDS, deb, dd, ett, ed):
    dmark = "OK" if dok == gt else "XX"
    emark = "OK" if eok == gt else "XX"
    flag = "  <<<" if dmark != emark else ""
    print(f"{('STORE' if gt else 'skip'):>5} {ds:>5.2f}{dmark:>3} {es:>5.2f}{emark:>3}  {text[:60]}{flag}")
    if dok != gt and eok != gt:
        both_miss.append((gt, text))
    elif dok != gt:
        deb_miss.append((gt, text))
    elif eok != gt:
        ett_miss.append((gt, text))

def dump(title, items):
    print(f"\n{title} ({len(items)}):")
    for gt, t in items:
        print(f"  [{'STORE' if gt else 'skip'}] {t}")

dump("DeBERTa misses (ettin correct)", deb_miss)
dump("ettin-150m misses (DeBERTa correct)", ett_miss)
dump("BOTH miss (likely genuinely ambiguous / label-debatable)", both_miss)
print(f"\nDeBERTa total wrong: {sum(d!=g for d,g in zip(dd,labels))}/64 | "
      f"ettin total wrong: {sum(e!=g for e,g in zip(ed,labels))}/64")
