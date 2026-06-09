"""One-off: measure ettin cross-encoder inference latency CPU vs CUDA for the two
real call shapes in the hot path. Run once per device (fresh process so the
cross_encoder_model singleton loads on the requested device):
    python _ettin_cpu_bench.py --device cpu
    python _ettin_cpu_bench.py --device cuda
Reports per-call latency (NOT cold load — that's prewarmed at boot and hidden)."""
import argparse
import statistics
import time

import cross_encoder_model
from config import CROSS_ENCODER_MODEL

# Real batch shapes:
#  - rerank  (memory.py:215): up to MEMORY_RERANK_TOP_N=15 (query, passage) pairs, EVERY turn
#  - salience (salience_layer.py:223): 6 (hypothesis, text) pairs, COMMIT turns only
QUERY = "What do you remember about what I like to eat?"
PASSAGES = [
    "The user's name is Jamie.",
    "Jamie is a freelance fashion designer.",
    "The user loves fresh fruit, especially strawberries.",
    "The user's blood type is O and they will not donate organs.",
    "We talked about cheese earlier today.",
    "The weather in Berlin was warm and sunny.",
    "Jamie views this project as part of their soul.",
    "The user asked about the local time in Potsdam.",
    "The user prefers aggressive iteration when building.",
    "Jamie mentioned a strawberry field design concept.",
    "The user runs three frontier LLMs plus one human.",
    "We were testing the memory and recall system.",
    "The user dislikes flattery and wants honest calibration.",
    "Jamie built the host PC and torture-tested it.",
    "The assistant speaks through text-to-speech.",
]
RERANK_PAIRS = [[QUERY, p] for p in PASSAGES]            # 15 pairs

HYPOS = [
    "This states who the user or their family is — their name, job, background, or relationships.",
    "This states a lasting preference, like, dislike, habit, or value of the user.",
    "This is about the user's long-term project, goal, or ongoing work.",
    "This expresses something emotionally significant or a major life event for the user.",
    "This states a health condition, allergy, dietary restriction, or other lasting personal constraint.",
    "This is a transient, time-bound, or throwaway detail that is not worth remembering.",
]
SAL_TEXT = "The user loves fresh fruit, especially strawberries."
SALIENCE_PAIRS = [[h, SAL_TEXT] for h in HYPOS]          # 6 pairs


def bench(model, pairs, label, iters=40):
    # warmup
    for _ in range(5):
        model.predict(pairs, show_progress_bar=False)
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        model.predict(pairs, show_progress_bar=False)
        samples.append((time.perf_counter() - t) * 1000.0)  # ms
    samples.sort()
    med = statistics.median(samples)
    p90 = samples[int(len(samples) * 0.9)]
    print(f"  {label:24s} n={len(pairs):2d} pairs  "
          f"median={med:7.1f} ms  p90={p90:7.1f} ms  "
          f"min={samples[0]:6.1f}  max={samples[-1]:6.1f}")
    return med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    args = ap.parse_args()

    t0 = time.perf_counter()
    model = cross_encoder_model.get_model(CROSS_ENCODER_MODEL, args.device)
    load_s = time.perf_counter() - t0
    if model is None:
        print(f"[{args.device}] model unavailable — load failed.")
        return
    print(f"\n=== ettin-150m on {args.device.upper()} "
          f"(cold load {load_s:.1f}s; load is prewarmed at boot, excluded below) ===")
    bench(model, RERANK_PAIRS,   "rerank (every turn)")
    bench(model, SALIENCE_PAIRS, "salience (commit turns)")


if __name__ == "__main__":
    main()
