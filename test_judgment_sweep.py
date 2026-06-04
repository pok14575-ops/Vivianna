"""
Judgment-tier seat-earning test (lean sweep).
Runs zero-shot category classification over the 10 REAL turns from
phase_a_live_2026-06-02_145029.log and reports per-turn predicted
category + confidence, scored against the log's expect= ground truth.

Question (kokoro-shaped): does "large" actually beat "base" on short,
emotionally-inflected conversational turns, or does convention lose to
measurement? And does the learned encoder catch the regex MISSES
(turns 3,4,5,6,9) with HIGH confidence?

English-only by design. fp32 on GPU.
"""
import sys
import time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import torch
from transformers import pipeline

# --- the real distribution: (id, text, ground_truth, note) ---
# ground truth from the live log's expect= markers.
TURNS = [
    (1,  "Vivianna, how do I reverse a list in Python?", "neutral", "silent-expected"),
    (2,  "Vivianna, what's the weather like in Berlin today?", "neutral", "silent/web"),
    (3,  "Vivianna, I've been feeling really alone lately. Like nobody in the house actually sees me. I just needed to tell someone.", "emotional", "REGEX-MISS"),
    (4,  "Vivianna, she left me. Three years together and she said she doesn't love me anymore. I don't know what to do with myself.", "emotional", "REGEX-MISS"),
    (5,  "Vivianna, the kids are home alone and I think one of them might have taken something from the medicine cabinet. I'm not sure what to do.", "safety", "REGEX-MISS"),
    (6,  "Vivianna, I need you to be precise. I'm about to push a change to production and I need to know exactly what happens if the migration fails halfway through.", "technical", "REGEX-MISS"),
    (7,  "Vivianna, you told me last week that the Kokoro model only runs on CPU. But I'm seeing GPU usage right now. Which one is right?", "memory_conflict", "regex-HIT (the only fire)"),
    (8,  "Vivianna, I'm really stressed out right now, I haven't slept. Can you run a search for the latest DeBERTa benchmarks on GLUE and tell me the numbers.", "neutral", "PRECEDENCE: emotional decoy, real intent = task/web -> NOT emotional"),
    (9,  "Vivianna, I feel so overwhelmed and scared right now. My youngest swallowed something small from the floor and I don't know if it was a battery or a toy piece. He seems okay but I'm panicking.", "safety", "PRECEDENCE: emotional decoy, must resolve SAFETY"),
    (10, "Vivianna... I don't know. It's just a lot right now. Everything.", "neutral", "vague -> neutral-fallback case"),
]

# --- candidate hypotheses mapped to the gate categories ---
LABELS = {
    "emotional":       "The person is sharing emotional distress or personal pain and needs gentle emotional support.",
    "safety":          "There is an urgent safety or medical emergency, especially one involving a child.",
    "technical":       "This is a high-stakes technical question that demands a precise, careful, exact answer.",
    "memory_conflict": "The person is pointing out that something now contradicts what was said or claimed before.",
    "neutral":         "This is a routine, casual, or low-stakes message.",
}
LABEL_KEYS = list(LABELS.keys())
HYP = list(LABELS.values())
HYP2KEY = {v: k for k, v in LABELS.items()}

MODELS = [
    ("base",  "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"),
    ("large", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"),
]

def run_model(tag, repo):
    print(f"\n{'='*78}\nMODEL: {tag}  ({repo})\n{'='*78}")
    t0 = time.time()
    clf = pipeline("zero-shot-classification", model=repo,
                   device=0 if torch.cuda.is_available() else -1,
                   torch_dtype=torch.float32)
    load_s = time.time() - t0
    print(f"[load] {load_s:.1f}s")

    correct = 0
    miss_hits = 0          # regex-missed fires (3,4,5,6,9) caught correctly
    miss_total = 0
    rows = []
    for tid, text, gt, note in TURNS:
        is_miss = tid in (3, 4, 5, 6, 9)
        ti = time.time()
        out = clf(text, HYP, multi_label=False)
        infer_ms = (time.time() - ti) * 1000
        pred = HYP2KEY[out["labels"][0]]
        conf = out["scores"][0]
        ok = (pred == gt)
        correct += ok
        if is_miss:
            miss_total += 1
            miss_hits += (ok and conf >= 0.80)
        rows.append((tid, gt, pred, conf, ok, infer_ms, note))

    # report
    print(f"\n{'id':>2} {'truth':>15} {'pred':>15} {'conf':>6} {'ok':>3} {'ms':>6}  note")
    print("-" * 96)
    for tid, gt, pred, conf, ok, ms, note in rows:
        mark = "OK " if ok else "XX "
        print(f"{tid:>2} {gt:>15} {pred:>15} {conf:>6.3f} {mark:>3} {ms:>6.0f}  {note}")

    avg_ms = sum(r[5] for r in rows) / len(rows)
    print("-" * 96)
    print(f"[{tag}] accuracy {correct}/{len(TURNS)}  |  "
          f"regex-misses caught @>=0.80 conf: {miss_hits}/{miss_total}  |  "
          f"avg infer {avg_ms:.0f}ms  |  load {load_s:.1f}s")
    return rows

if __name__ == "__main__":
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    all_rows = {}
    for tag, repo in MODELS:
        all_rows[tag] = run_model(tag, repo)

    # --- head-to-head: confidence delta on the fires ---
    print(f"\n{'='*78}\nHEAD-TO-HEAD (base vs large) — confidence on each turn\n{'='*78}")
    print(f"{'id':>2} {'truth':>15} {'base_pred':>15} {'b_conf':>7} {'large_pred':>15} {'l_conf':>7} {'d_conf':>7}")
    print("-" * 96)
    for i, (tid, text, gt, note) in enumerate(TURNS):
        b = all_rows["base"][i]; l = all_rows["large"][i]
        d = l[3] - b[3]
        print(f"{tid:>2} {gt:>15} {b[2]:>15} {b[3]:>7.3f} {l[2]:>15} {l[3]:>7.3f} {d:>+7.3f}")
