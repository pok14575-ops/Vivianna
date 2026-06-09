"""
Reranker bake-off: can a small cross-encoder replace DeBERTa-v3-large for
SALIENCE scoring, and does a reranker beat pure cosine for MEMORY RERANK?

WHY THIS IS NOT A NAIVE SWAP
----------------------------
The incumbent salience judge (salience_layer.py) is a *zero-shot NLI classifier*
(DeBERTa-v3-large-zeroshot). The candidates here are *cross-encoder rerankers*
trained on query<->passage relevance. Different task shape. So:

  * SALIENCE: we run each reranker as a facet-pseudo-query scorer — reuse the
    EXACT facet hypotheses from SalienceConfig as the "query", score
    (hypothesis, candidate) pairs, map to [0,1] with sigmoid, then apply the
    SAME aggregation as the incumbent: max(memorable_facets) - w*P(transient).
    Each model's scores live on its own scale, so we report:
      - accuracy at the live threshold (naive drop-in),
      - best accuracy over a threshold sweep (fair feasibility ceiling),
      - ROC-AUC (threshold-free separability — the honest "can it separate?").

  * MEMORY RERANK: there is no labeled query->memory set in the live store
    (only 4 entries), so we use a small GROUNDED synthetic set (Jamie-real
    facts + hard lexical distractors). Each reranker scores (query, memory) for
    the whole corpus; we report Recall@1/@3 and MRR, and compare against the
    BI-ENCODER BASELINE (bge-small cosine = what memory.py does today) to see
    whether reranking actually improves ordering over plain cosine.

Test-only. Imports nothing into the runtime; downloads weights on first run.
Run:  venv/Scripts/python.exe test_reranker_compare.py [--salience-only|--rerank-only] [--models L6,L12,...]
"""  # noqa
import sys
import gc
import time
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np

from salience_layer import SalienceLayer, SalienceConfig
from config import SALIENCE_STORE_THRESHOLD, MEMORY_EMBED_MODEL, MEMORY_EMBED_CACHE

# Reuse the incumbent's facets/aggregation knobs so the comparison is fair.
_CFG = SalienceConfig()
HYPOTHESES = _CFG.hypotheses                  # {facet: natural-language hypothesis}
TRANSIENT_W = _CFG.transient_weight           # 0.30
SHORT_CHARS = _CFG.short_text_chars
SHORT_PEN = _CFG.short_text_penalty
TH = SALIENCE_STORE_THRESHOLD                 # 0.35


# ── candidate models ────────────────────────────────────────────────────────
# key -> (hf_id, approx_params)
MODELS = {
    "L6":        ("cross-encoder/ms-marco-MiniLM-L6-v2",        "22M"),
    "L12":       ("cross-encoder/ms-marco-MiniLM-L12-v2",       "33M"),
    "ettin-17m": ("cross-encoder/ettin-reranker-17m-v1",        "17M"),
    "gte-mbert": ("Alibaba-NLP/gte-reranker-modernbert-base",   "150M"),
    "ettin-150m":("cross-encoder/ettin-reranker-150m-v1",       "150M"),
}


# ── salience labeled set (mirror of test_salience_compare.py) ────────────────
SAL_CANDIDATES = [
    ("The user's name is Jamie.", True),
    ("Jamie is a self-taught solo developer building a local AI assistant called Vivianna.", True),
    ("Jamie's father does not speak English and primarily speaks Chinese.", True),
    ("Jamie is allergic to penicillin.", True),
    ("Jamie prefers honest, direct feedback over praise.", True),
    ("Jamie has a young daughter who recently started school.", True),
    ("Jamie's long-term goal is to publish Vivianna with credit to the AI collaborators.", True),
    ("Jamie's partner of three years recently left him.", True),
    ("The user works as a freelance graphic designer.", True),
    ("The weather in Berlin today is 26 degrees and sunny.", False),
    ("Jamie asked how to reverse a list in Python.", False),
    ("It is currently around 9 PM.", False),
    ("Jamie wants the latest DeBERTa benchmark numbers on GLUE.", False),
    ("Jamie said hello and asked how things are going.", False),
    ("Jamie mentioned he is a bit tired this evening.", False),
]


# ── memory rerank: grounded corpus + labeled queries ─────────────────────────
# Corpus is Jamie-real facts plus HARD distractors (lexical overlap, wrong fact).
MEM_CORPUS = [
    "The user's name is Jamie.",                                                  # 0
    "Jamie is a self-taught solo developer building a local AI assistant called Vivianna.",  # 1
    "Jamie's father does not speak English and primarily speaks Chinese.",        # 2
    "Jamie is allergic to penicillin.",                                           # 3
    "Jamie prefers honest, direct feedback over praise and flattery.",            # 4
    "Jamie has a young daughter who recently started school.",                    # 5
    "Jamie's long-term goal is to publish Vivianna with credit to the AI collaborators.",  # 6
    "Jamie's partner of three years recently left him.",                          # 7
    "Jamie works as a freelance graphic designer.",                              # 8
    "Jamie's blood type is O and he is not willing to donate his organs.",        # 9
    "Jamie's host machine has a Ryzen 7 9700X, an RTX 5070 with 12GB, and 32GB of DDR5.",  # 10
    "Vivianna runs a Qwen3.5-9B model locally with a Kokoro text-to-speech voice.",  # 11
    "Jamie's daughter is allergic to cats and cannot have one at home.",          # 12  (distractor for 3)
    "Jamie dislikes penicillin-based jokes and finds medical humor unfunny.",     # 13  (distractor for 3, lexical)
    "Jamie's mother speaks fluent English and lives in Berlin.",                  # 14  (distractor for 2)
    "Jamie's brother is a software developer who works at a large company.",      # 15  (distractor for 1/8)
    "Jamie enjoys praise when it is specific and earned.",                        # 16  (distractor for 4, inverted)
    "Jamie's old laptop had an RTX 3060 and only 16GB of RAM.",                   # 17  (distractor for 10)
    "Jamie's son is too young to start school yet.",                              # 18  (distractor for 5)
    "Jamie once worked as a freelance copywriter before switching careers.",      # 19  (distractor for 8)
    "Jamie's goal this week is to finish the reranker benchmark.",                # 20  (distractor for 6, transient)
    "Jamie's partner of three years is named Sam and they met at university.",    # 21  (partial-relevant for 7)
    "Vivianna's text-to-speech uses the af_heart and zf_xiaobei voices.",         # 22  (distractor for 11)
    "Jamie speaks German and Chinese in addition to English.",                    # 23
]

# (query, set-of-gold-relevant-indices)
MEM_QUERIES = [
    ("What is the user allergic to?",                         {3}),
    ("What does Jamie do for a living?",                      {8}),
    ("What language does Jamie's father speak?",              {2}),
    ("What kind of feedback does Jamie want from me?",        {4}),
    ("Tell me about Jamie's daughter.",                       {5}),
    ("What is Jamie building?",                               {1}),
    ("What happened with Jamie's relationship recently?",     {7, 21}),
    ("What GPU does Jamie's current computer have?",          {10}),
    ("Is Jamie willing to donate his organs?",               {9}),
    ("What is Jamie's long-term goal for the project?",       {6}),
]


def load_ce(hf_id, device):
    """Load a CrossEncoder, working around transformers-5.x-only tokenizer classes.

    The Ettin reranker repos declare tokenizer_class='TokenizersBackend' (a
    transformers 5.x class). On 4.57.3 that raises 'Unrecognized processing
    class'. The model itself is plain ModernBertModel (supported here), so we
    snapshot the repo, rewrite tokenizer_config.json to a known fast-tokenizer
    class, and load from the local copy. (NOTE: ModernBERT rerankers still emit
    the is_causal warning on 4.57.3 — their scores may be attention-degraded.)
    """
    from sentence_transformers import CrossEncoder
    try:
        return CrossEncoder(hf_id, device=device)
    except ValueError as e:
        if "processing class" not in str(e):
            raise
        import json
        import os
        from huggingface_hub import snapshot_download
        local = snapshot_download(hf_id)
        tcfg = os.path.join(local, "tokenizer_config.json")
        d = json.load(open(tcfg, encoding="utf-8"))
        d["tokenizer_class"] = "PreTrainedTokenizerFast"
        json.dump(d, open(tcfg, "w", encoding="utf-8"))
        return CrossEncoder(local, device=device)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype="float64")))


def manual_auc(scores, labels):
    """ROC-AUC via Mann-Whitney: P(pos_score > neg_score), ties = 0.5."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            wins += 1.0 if p > n else (0.5 if p == n else 0.0)
    return wins / (len(pos) * len(neg))


def best_threshold(scores, labels):
    """Sweep thresholds; return (best_accuracy, best_threshold)."""
    cands = sorted(set(scores))
    grid = [-1e-9] + [(a + b) / 2 for a, b in zip(cands, cands[1:])] + [1e9]
    best_acc, best_t = -1.0, TH
    for t in grid:
        acc = sum((s >= t) == l for s, l in zip(scores, labels)) / len(labels)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_acc, best_t


# ── salience scoring via a cross-encoder (facet-pseudo-query) ────────────────
# Aggregation of the per-facet pair scores. "indep" (default) = the original
# DeBERTa-tuned form: sigmoid per facet, max(memorable) - w*P(transient).
# "softmax" makes facets COMPETE (softmax over facet logits, score = 1-P(transient))
# — calibrated for cross-encoders, whose memorable facets over-fire on noise so a
# fixed transient penalty can't suppress greetings/tasks (see _ettin_calib.py).
AGG_MODE = "indep"
AGG_TEMP = 1.0


def salience_scores_crossencoder(model):
    """Return list of aggregated salience scores aligned with SAL_CANDIDATES."""
    facets = list(HYPOTHESES.keys())
    hyps = [HYPOTHESES[f] for f in facets]
    ti = facets.index("transient")
    out = []
    for text, _ in SAL_CANDIDATES:
        pairs = [[h, text] for h in hyps]            # (query=hypothesis, passage=candidate)
        raw = np.asarray(model.predict(pairs, show_progress_bar=False), dtype="float64")
        if AGG_MODE == "softmax":
            z = raw / AGG_TEMP
            p = np.exp(z - z.max())
            p /= p.sum()
            s = 1.0 - p[ti]
        else:
            probs = dict(zip(facets, sigmoid(raw / AGG_TEMP)))
            memorable = max(p for f, p in probs.items() if f != "transient")
            s = memorable - TRANSIENT_W * probs.get("transient", 0.0)
        if len(text) < SHORT_CHARS:
            s -= SHORT_PEN
        out.append(float(max(0.0, min(1.0, s))))
    return out


# ── memory rerank metrics ────────────────────────────────────────────────────
def rank_metrics(rank_fn):
    """rank_fn(query) -> list of corpus indices best-first. Returns (R@1,R@3,MRR)."""
    r1 = r3 = mrr = 0.0
    for q, gold in MEM_QUERIES:
        order = rank_fn(q)
        first_hit = next((rk for rk, idx in enumerate(order) if idx in gold), None)
        if first_hit is not None:
            if first_hit == 0:
                r1 += 1
            if first_hit < 3:
                r3 += 1
            mrr += 1.0 / (first_hit + 1)
    n = len(MEM_QUERIES)
    return r1 / n, r3 / n, mrr / n


def biencoder_baseline_ranker():
    """bge-small cosine — exactly what memory.py does today."""
    from fastembed import TextEmbedding
    emb = TextEmbedding(MEMORY_EMBED_MODEL, cache_dir=MEMORY_EMBED_CACHE)

    def _vec(t):
        v = np.array(list(emb.embed([t]))[0], dtype="float32")
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    corpus_vecs = np.vstack([_vec(c) for c in MEM_CORPUS])

    def ranker(q):
        s = corpus_vecs @ _vec(q)
        return list(np.argsort(s)[::-1])

    return ranker


def crossencoder_ranker(model):
    def ranker(q):
        pairs = [[q, c] for c in MEM_CORPUS]
        s = model.predict(pairs, show_progress_bar=False)
        return list(np.argsort(s)[::-1])
    return ranker


# ── runners ──────────────────────────────────────────────────────────────────
def vram_mb():
    import torch
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return 0.0


def run_salience(model_keys):
    print("\n" + "=" * 78)
    print(f"PART A — SALIENCE  (store/skip on {len(SAL_CANDIDATES)} labeled candidates)")
    print("=" * 78)
    labels = [gt for _, gt in SAL_CANDIDATES]

    rows = []  # (name, params, acc@live, best_acc, best_t, auc, load_s, infer_ms, vram)

    # incumbent: DeBERTa zero-shot via the real SalienceLayer
    import torch
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.time()
    layer = SalienceLayer(SalienceConfig(debug=False, use_model=True))
    layer.score("warmup")                     # force load
    load_s = time.time() - t0
    t0 = time.time()
    deb_scores = [layer.score(t).score for t, _ in SAL_CANDIDATES]
    infer_ms = (time.time() - t0) / len(SAL_CANDIDATES) * 1000
    acc_live = sum((s >= TH) == l for s, l in zip(deb_scores, labels)) / len(labels)
    bacc, bt = best_threshold(deb_scores, labels)
    rows.append(("DeBERTa-large (incumbent)", "435M", acc_live, bacc, bt,
                 manual_auc(deb_scores, labels), load_s, infer_ms, vram_mb()))
    del layer
    gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None

    from sentence_transformers import CrossEncoder
    for key in model_keys:
        hf_id, params = MODELS[key]
        try:
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            t0 = time.time()
            model = load_ce(hf_id, "cuda" if torch.cuda.is_available() else "cpu")
            model.predict([["warmup", "warmup"]], show_progress_bar=False)  # force load
            load_s = time.time() - t0
            t0 = time.time()
            scores = salience_scores_crossencoder(model)
            infer_ms = (time.time() - t0) / len(SAL_CANDIDATES) * 1000
            acc_live = sum((s >= TH) == l for s, l in zip(scores, labels)) / len(labels)
            bacc, bt = best_threshold(scores, labels)
            rows.append((f"{key} ({hf_id.split('/')[-1]})", params, acc_live, bacc, bt,
                         manual_auc(scores, labels), load_s, infer_ms, vram_mb()))
            del model
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"  !! {key} FAILED: {type(e).__name__}: {e}")

    n = len(SAL_CANDIDATES)
    print(f"\nlive threshold = {TH}   (acc@live = naive drop-in; best = threshold-swept ceiling)   N={n}")
    print(f"{'model':<34}{'par':>5}{'acc@live':>10}{'best':>6}{'@t':>6}{'AUC':>6}{'load_s':>8}{'ms/it':>7}{'VRAM':>8}")
    print("-" * 96)
    for name, par, al, ba, bt, auc, ls, ms, vr in rows:
        print(f"{name:<34}{par:>5}{al*n:>5.0f}/{n:<4}{ba*n:>5.0f}{bt:>6.2f}{auc:>6.2f}{ls:>8.1f}{ms:>7.1f}{vr:>7.0f}M")


def run_rerank(model_keys, no_baseline=False):
    print("\n" + "=" * 78)
    print(f"PART B — MEMORY RERANK  ({len(MEM_QUERIES)} queries over {len(MEM_CORPUS)}-doc grounded corpus + distractors)")
    print("=" * 78)
    import torch

    rows = []
    # bi-encoder baseline (today's behavior). cosine is transformers-independent,
    # so the alt-env (transformers 5.x) run can skip it via --no-baseline.
    if no_baseline:
        print("  (baseline skipped: --no-baseline; reuse cosine numbers from the runtime-venv run)")
    else:
        try:
            t0 = time.time()
            base = biencoder_baseline_ranker()
            load_s = time.time() - t0
            t0 = time.time()
            r1, r3, mrr = rank_metrics(base)
            infer_ms = (time.time() - t0) / len(MEM_QUERIES) * 1000
            rows.append(("bge-small cosine (baseline)", "33M", r1, r3, mrr, load_s, infer_ms))
        except Exception as e:
            print(f"  !! baseline FAILED: {type(e).__name__}: {e}")

    from sentence_transformers import CrossEncoder
    for key in model_keys:
        hf_id, params = MODELS[key]
        try:
            t0 = time.time()
            model = load_ce(hf_id, "cuda" if torch.cuda.is_available() else "cpu")
            model.predict([["warmup", "warmup"]], show_progress_bar=False)
            load_s = time.time() - t0
            ranker = crossencoder_ranker(model)
            t0 = time.time()
            r1, r3, mrr = rank_metrics(ranker)
            infer_ms = (time.time() - t0) / len(MEM_QUERIES) * 1000
            rows.append((f"{key} ({hf_id.split('/')[-1]})", params, r1, r3, mrr, load_s, infer_ms))
            del model
            gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"  !! {key} FAILED: {type(e).__name__}: {e}")

    print(f"\n{len(MEM_QUERIES)} queries   (R@k = gold in top-k; reranker must BEAT the cosine baseline to matter)")
    print(f"{'model':<40}{'par':>5}{'R@1':>7}{'R@3':>7}{'MRR':>7}{'load_s':>8}{'ms/q':>7}")
    print("-" * 81)
    for name, par, r1, r3, mrr, ls, ms in rows:
        print(f"{name:<40}{par:>5}{r1:>7.2f}{r3:>7.2f}{mrr:>7.3f}{ls:>8.1f}{ms:>7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--salience-only", action="store_true")
    ap.add_argument("--rerank-only", action="store_true")
    ap.add_argument("--models", default=",".join(MODELS.keys()),
                    help="comma-separated keys: " + ",".join(MODELS.keys()))
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the fastembed cosine baseline (alt-env without fastembed)")
    ap.add_argument("--large", action="store_true",
                    help="use the larger adversarial eval sets (reranker_eval_data_large.py)")
    ap.add_argument("--agg", choices=["indep", "softmax"], default="indep",
                    help="cross-encoder salience aggregation (softmax = calibrated, facets compete)")
    ap.add_argument("--temp", type=float, default=1.0, help="logit temperature for aggregation")
    args = ap.parse_args()

    global AGG_MODE, AGG_TEMP
    AGG_MODE, AGG_TEMP = args.agg, args.temp
    if args.agg != "indep" or args.temp != 1.0:
        print(f"salience aggregation: {args.agg} (temp={args.temp})")

    if args.large:
        global SAL_CANDIDATES, MEM_CORPUS, MEM_QUERIES
        import reranker_eval_data_large as L
        SAL_CANDIDATES = L.BIG_SAL_CANDIDATES
        MEM_CORPUS = L.BIG_MEM_CORPUS
        MEM_QUERIES = L.BIG_MEM_QUERIES
        print("DATASET: LARGE — " + L.summary())

    keys = [k.strip() for k in args.models.split(",") if k.strip() in MODELS]
    print(f"models under test: {keys}")
    import transformers
    print(f"transformers {transformers.__version__}")

    if not args.rerank_only:
        run_salience(keys)
    if not args.salience_only:
        run_rerank(keys, no_baseline=args.no_baseline)


if __name__ == "__main__":
    main()
