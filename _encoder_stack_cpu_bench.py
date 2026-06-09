"""One-off: CPU (and CUDA) latency for the two NOT-yet-wired encoder-stack-v2 models,
to decide whether they can live on CPU. Tests the real shapes:
  - go_emotions: ONE short user utterance, one pass/turn
  - deberta-small-long-nli (grounding): one (premise, hypothesis) pass, but at several
    PREMISE lengths — the point is to show CPU cost is driven by context length, not size.
Run: python _encoder_stack_cpu_bench.py --device cpu   (and --device cuda)
"""
import argparse, statistics, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

NLI_MODEL = "tasksource/deberta-small-long-nli"
EMO_MODEL = "SamLowe/roberta-base-go_emotions"


def _params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def _time(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        s.append((time.perf_counter() - t) * 1000.0)
    s.sort()
    return statistics.median(s), s[int(len(s) * 0.9)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("cuda not available"); return
    print(f"\n=== encoder-stack-v2 latency on {dev.upper()} (per-call, load excluded) ===")

    # ---- grounding NLI: vary the premise length ----
    tok = AutoTokenizer.from_pretrained(NLI_MODEL)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL).to(dev).eval()
    print(f"  deberta-small-long-nli : {_params(nli):.0f}M params")
    hyp = "The user is allergic to peanuts."
    sent = ("Jamie is a freelance fashion designer who loves fresh fruit, especially "
            "strawberries, and views this project as part of their soul. ")
    for approx_tokens in (32, 256, 1024, 1680):
        # build a premise of ~approx_tokens by repeating, then truncate exactly
        premise = sent * (approx_tokens // 20 + 1)
        enc = tok(premise, hyp, return_tensors="pt", truncation=True, max_length=approx_tokens)
        n_tok = enc["input_ids"].shape[1]
        enc = {k: v.to(dev) for k, v in enc.items()}
        def run(enc=enc):
            with torch.no_grad():
                nli(**enc).logits.softmax(-1)
            if dev == "cuda":
                torch.cuda.synchronize()
        med, p90 = _time(run)
        print(f"    grounding  premise~{n_tok:5d} tok   median={med:8.1f} ms  p90={p90:8.1f} ms")

    # ---- go_emotions: one short utterance ----
    emo = pipeline("text-classification", model=EMO_MODEL, top_k=None,
                   device=0 if dev == "cuda" else -1)
    nparams = _params(emo.model)
    print(f"  roberta-base-go_emotions: {nparams:.0f}M params")
    utt = "I'm really worried this whole thing is going to fall apart."
    def run_emo():
        emo(utt)
    med, p90 = _time(run_emo)
    print(f"    emotion    short utterance      median={med:8.1f} ms  p90={p90:8.1f} ms")


if __name__ == "__main__":
    main()
