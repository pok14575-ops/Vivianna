"""fp16-vs-fp32 parity + cost bench for the two not-yet-wired judges.

WHY this exists: the handoff asserted "fp16 precision loss is immaterial for 3-class /
28-way." Kokoro int8 taught us not to trust that assertion (int8 was 3.8x worse than fp32).
fp16 is a much milder perturbation than int8, but we MEASURE rather than assume. The gate is
VRAM (~800 MB margin on the 12 GB card with ettin already resident), and the question fp16
raises is: does halving the weights / activation spike change any DECISION the judges make?

Three axes, fp32 and fp16 side by side, CUDA only (fp16 on CPU is meaningless):
  1. CORRECTNESS  - argmax/top-label agreement + softmax/sigmoid prob deltas vs fp32 ref.
  2. VRAM         - weight footprint + peak (incl. the 1680-tok activation spike).
  3. LATENCY      - 1680-tok grounding pass + short emotion pass.

Run:  venv_tf5\Scripts\python.exe _encoder_fp16_fp32_bench.py
"""
import argparse, gc, statistics, time
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

NLI_MODEL = "tasksource/deberta-small-long-nli"
EMO_MODEL = "SamLowe/roberta-base-go_emotions"

# ---- NLI parity set: claim-vs-memory pairs in the REAL grounding shape ----
# (premise = stored memory / context, hypothesis = new claim). Mix of clear and
# near-boundary cases so a precision flip has somewhere to show up.
NLI_PAIRS = [
    ("The user is allergic to peanuts.", "The user can safely eat peanuts."),          # contradict
    ("The user is vegan and avoids all animal products.", "The user loves steak."),    # contradict
    ("The user lives in Berlin, Germany.", "The user lives in Germany."),              # entail
    ("The user is a freelance fashion designer.", "The user works in fashion."),       # entail
    ("The user has a wife named Vivianna.", "The user enjoys hiking on weekends."),     # neutral
    ("The user's father speaks Mandarin Chinese.", "The user speaks Chinese."),        # neutral-ish
    ("The user prefers tea over coffee.", "The user dislikes coffee."),                # neutral/near
    ("The user is afraid of spiders.", "The user keeps a pet tarantula."),             # contradict-ish
    ("The user works night shifts this week.", "The user is awake during the night."), # entail
    ("The user is learning German.", "The user is fluent in German."),                 # neutral/near
    ("The user owns an RTX 5070 graphics card.", "The user has a 12 GB GPU."),         # entail (world)
    ("The user dislikes loud environments.", "The user enjoys nightclubs."),           # contradict
    ("The user had cereal for breakfast.", "The user skipped breakfast."),             # contradict
    ("The user is a parent of two children.", "The user has a family."),               # entail
    ("The user codes in Python.", "The user has never programmed."),                   # contradict
    ("The user mentioned being tired today.", "The user feels energetic today."),      # contradict/near
]

EMO_UTTS = [
    "I'm really worried this whole thing is going to fall apart.",
    "This is amazing, I can't believe it actually worked!",
    "I'm so tired, I just want to sleep for a week.",
    "Why does this keep breaking, it's infuriating.",
    "Thank you so much, that genuinely means a lot to me.",
    "I'm not sure how I feel about this anymore.",
    "That's hilarious, I can't stop laughing.",
    "I feel really alone tonight.",
    "Let's just get this done and move on.",
    "I'm proud of how far this project has come.",
    "Honestly that's kind of disgusting.",
    "I'm nervous about the demo tomorrow.",
]


def _params(m):
    return sum(p.numel() for p in m.parameters()) / 1e6


def _time(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        s.append((time.perf_counter() - t) * 1000.0)
    s.sort()
    return statistics.median(s), s[int(len(s) * 0.9)]


def _free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _nli_probs(model, tok, dev, max_len=1680):
    """Return (N,3) prob tensor on CPU for the parity set, padded/truncated to a fixed
    premise budget so shapes match the real read-time grounding pass."""
    out = []
    with torch.no_grad():
        for premise, hyp in NLI_PAIRS:
            enc = tok(premise, hyp, return_tensors="pt", truncation=True, max_length=max_len)
            enc = {k: v.to(dev) for k, v in enc.items()}
            p = model(**enc).logits.float().softmax(-1).cpu()
            out.append(p)
    return torch.cat(out, 0)


def _emo_probs(model, tok, dev):
    """Multi-label sigmoid probs (N, num_labels) on CPU."""
    out = []
    with torch.no_grad():
        for utt in EMO_UTTS:
            enc = tok(utt, return_tensors="pt", truncation=True, max_length=64)
            enc = {k: v.to(dev) for k, v in enc.items()}
            p = model(**enc).logits.float().sigmoid().cpu()
            out.append(p)
    return torch.cat(out, 0)


def _load(model_id, dtype, dev):
    return AutoModelForSequenceClassification.from_pretrained(
        model_id, torch_dtype=dtype).to(dev).eval()


def run_model(name, model_id, probe, dev, long_premise_len):
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"\n================  {name}  ================")

    # ---- fp32 reference ----
    _free()
    m32 = _load(model_id, torch.float32, dev)
    nparams = _params(m32)
    w32 = torch.cuda.memory_allocated() / 1e6
    ref = probe(m32, tok, dev)
    peak32 = torch.cuda.max_memory_allocated() / 1e6
    print(f"  params: {nparams:.0f}M   |  fp32 weights ~{w32:.0f} MB   peak ~{peak32:.0f} MB   "
          f"(activation spike ~{peak32 - w32:.0f} MB)")
    del m32; _free()

    # ---- fp16 ----
    m16 = _load(model_id, torch.float16, dev)
    w16 = torch.cuda.memory_allocated() / 1e6
    got = probe(m16, tok, dev)
    peak16 = torch.cuda.max_memory_allocated() / 1e6
    print(f"  {'':24s}|  fp16 weights ~{w16:.0f} MB   peak ~{peak16:.0f} MB   "
          f"(activation spike ~{peak16 - w16:.0f} MB)")
    print(f"  VRAM SAVED by fp16: weights -{w32 - w16:.0f} MB   peak -{peak32 - peak16:.0f} MB")

    # ---- parity ----
    ref_arg = ref.argmax(-1)
    got_arg = got.argmax(-1)
    arg_agree = (ref_arg == got_arg).float().mean().item()
    abs_delta = (ref - got).abs()
    print(f"  PARITY  argmax/top-label agreement: {arg_agree*100:.1f}%   "
          f"({int((ref_arg==got_arg).sum())}/{len(ref_arg)})")
    print(f"          prob delta  max={abs_delta.max():.4f}  mean={abs_delta.mean():.5f}")
    # show any decision flips
    flips = (ref_arg != got_arg).nonzero(as_tuple=True)[0].tolist()
    if flips:
        print(f"  *** {len(flips)} DECISION FLIP(S) fp32->fp16:")
        for i in flips:
            print(f"        idx {i}: fp32 cls {ref_arg[i].item()} (p={ref[i].max():.3f}) "
                  f"-> fp16 cls {got_arg[i].item()} (p={got[i].max():.3f})")
    else:
        print("  *** no decision flips.")
    # worst single-prob disagreement (even without a flip)
    wi = abs_delta.max(-1).values.argmax().item()
    print(f"          worst row idx {wi}: fp32={ref[wi].tolist()}  fp16={got[wi].tolist()}")

    # ---- latency (fp16, the candidate; + fp32 for the long pass) ----
    if probe is None:
        pass
    del m16; _free()
    return arg_agree, flips


def latency_block(model_id, dev, max_len):
    tok = AutoTokenizer.from_pretrained(model_id)
    sent = ("Jamie is a freelance fashion designer who loves fresh fruit, especially "
            "strawberries, and views this project as part of their soul. ")
    premise = sent * (max_len // 20 + 1)
    hyp = "The user is allergic to peanuts."
    enc = tok(premise, hyp, return_tensors="pt", truncation=True, max_length=max_len)
    n_tok = enc["input_ids"].shape[1]
    print(f"\n  --- latency @ premise ~{n_tok} tok ---")
    for dtype, tag in ((torch.float32, "fp32"), (torch.float16, "fp16")):
        _free()
        m = _load(model_id, dtype, dev)
        e = {k: v.to(dev) for k, v in enc.items()}
        def run(e=e, m=m):
            with torch.no_grad():
                m(**e).logits.softmax(-1)
        med, p90 = _time(run)
        print(f"      {tag}: median={med:7.1f} ms   p90={p90:7.1f} ms")
        del m; _free()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-len", type=int, default=1680,
                    help="premise budget for the grounding pass / VRAM spike")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA required for fp16 bench."); return
    dev = "cuda"
    torch.cuda.init()
    print(f"GPU: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")
    print(f"NLI premise budget: {args.max_len} tok")

    run_model("deberta-small-long-nli (grounding, 3-class softmax)", NLI_MODEL,
              lambda m, t, d: _nli_probs(m, t, d, args.max_len), dev, args.max_len)
    latency_block(NLI_MODEL, dev, args.max_len)

    run_model("roberta-base-go_emotions (emotion, 28-way sigmoid)", EMO_MODEL,
              _emo_probs, dev, 64)

    print("\nDone. fp16 is acceptable iff: 0 decision flips on NLI (contradiction guard is "
          "the safety-critical path) AND VRAM saving is real. Read the flips, not just the %.")


if __name__ == "__main__":
    main()
