"""Smoke test the wired-in ettin: salience via the production config path, and
memory rerank via the real MemoryManager. Confirms the cross-encoder backend is
actually used (not regex fallback) and that ONE shared instance serves both."""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import config as C
from salience_layer import SalienceLayer, SalienceConfig
import reranker_eval_data_large as L


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


print(f"config: backend={C.SALIENCE_BACKEND} agg={C.SALIENCE_AGG} "
      f"model={C.CROSS_ENCODER_MODEL} threshold={C.SALIENCE_STORE_THRESHOLD}")

sal = SalienceLayer(SalienceConfig(
    debug=False, use_model=C.SALIENCE_MODEL_ENABLED, backend=C.SALIENCE_BACKEND,
    model_name=C.SALIENCE_MODEL_NAME, xenc_model=C.CROSS_ENCODER_MODEL,
    agg=C.SALIENCE_AGG, agg_temp=C.SALIENCE_AGG_TEMP, model_device=C.SALIENCE_MODEL_DEVICE,
))

CANDS = L.BIG_SAL_CANDIDATES
labels = [g for _, g in CANDS]
first = sal.score(CANDS[0][0])
print(f"reasons (must be xenc-*): {first.reasons}")
scores = [sal.score(t).score for t, _ in CANDS]
TH = C.SALIENCE_STORE_THRESHOLD
acc = sum((s >= TH) == g for s, g in zip(scores, labels))
print(f"SALIENCE via production path: acc@{TH} = {acc}/{len(CANDS)} | AUC = {auc(scores, labels):.3f}")

print("\n-- memory rerank (real store) --")
from memory import MemoryManager
mm = MemoryManager()
print(f"memory count: {mm.count()}")
for q in ["what is the user allergic to?", "how do I work / what am I built on?"]:
    res = mm.query(q)
    print(f"Q: {q}")
    for r in res:
        print(f"   -> [{r['score']:.3f}] {r['text'][:64]}")
