# grounding.py
"""Shared, lazy, fallback-safe 3-class NLI singleton — the encoder-stack-v2 "third piece".

ONE `deberta-small-long-nli` (142M) instance answers the one question ettin structurally
CANNOT: polarity. A relevance reranker scores "user is vegan" vs "user loves steak" HIGH
(topically related) — indistinguishable from agreement. NLI separates entail / neutral /
contradiction. Two consumers (both NOT yet wired — see grounding step 3):
  - READ-TIME grounding  (stabilizer.pre_generate): is a retrieved memory actually on-point
    for the answer (entailment), or does it conflict (contradiction)? Replaces cosine-only
    `memory_valid`.
  - WRITE-TIME contradiction guard (_auto_save_memory): does a new claim contradict an
    existing stored memory before we persist it?

Mirrors cross_encoder_model.py: thread-safe lazy load, prewarm daemon thread, any failure
leaves callers on their own fallback (cosine-only validity / no contradiction check).

Label order VERIFIED 2026-06-07 via _inspect_encoders.py (config.json id2label):
    0 = entailment, 1 = neutral, 2 = contradiction   (NLI ORDER == e/n/c: True)
max_position_embeddings = 1680 (tokenizer pinned to match).

Loaded in fp16 on CUDA (parity-proven: 0 decision flips vs fp32, _encoder_fp16_fp32_bench.py;
VRAM ~301MB peak vs 587MB fp32). fp32 on CPU (fp16 CPU ops are unsupported/slow).
"""
from __future__ import annotations

import threading

try:
    from config import GROUNDING_MODEL as _MODEL_DEFAULT
except Exception:
    _MODEL_DEFAULT = "tasksource/deberta-small-long-nli"
try:
    from config import GROUNDING_DEVICE as _DEVICE_DEFAULT
except Exception:
    _DEVICE_DEFAULT = "auto"
try:
    from config import GROUNDING_MAX_LEN as _MAX_LEN
except Exception:
    _MAX_LEN = 1680
try:
    from config import GROUNDING_DEBUG as _DEBUG
except Exception:
    _DEBUG = False

# Verified label indices (do NOT reorder without re-running _inspect_encoders.py).
ENTAIL, NEUTRAL, CONTRADICT = 0, 1, 2

_lock = threading.Lock()
_model = None            # (model, tokenizer) once loaded
_tok = None
_failed = False
_dev = None


def get_model(model_name: str = _MODEL_DEFAULT, device: str = _DEVICE_DEFAULT):
    """Return (model, tokenizer), loading on first call. (None, None) on failure.

    First successful call fixes the instance; later calls return it regardless of args."""
    global _model, _tok, _failed, _dev
    if _model is not None:
        return _model, _tok
    if _failed:
        return None, None
    with _lock:
        if _model is not None:
            return _model, _tok
        if _failed:
            return None, None
        try:
            import torch
            from transformers import (AutoTokenizer,
                                      AutoModelForSequenceClassification)

            dev = ("cuda" if torch.cuda.is_available() else "cpu") \
                if device == "auto" else device
            # fp16 only buys us anything on CUDA, and fp16 on CPU hits unsupported ops.
            dtype = torch.float16 if dev == "cuda" else torch.float32

            tok = AutoTokenizer.from_pretrained(model_name, model_max_length=_MAX_LEN)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name, dtype=dtype).to(dev).eval()

            # Guard the verified label order against a silently-different repo revision:
            # if it ever stops being e/n/c, fall back rather than invert the guard.
            id2 = {int(k): v.lower() for k, v in model.config.id2label.items()}
            if id2.get(ENTAIL) != "entailment" or id2.get(CONTRADICT) != "contradiction":
                raise ValueError(f"unexpected NLI label order {id2}; refusing to load "
                                 f"(would invert the contradiction guard)")

            _model, _tok, _dev = model, tok, dev
            if _DEBUG:
                print(f"[GROUND] loaded {model_name} on {dev} ({dtype})", flush=True)
            return _model, _tok
        except Exception as e:  # noqa: BLE001 — any failure -> callers fall back
            _failed = True
            _model = _tok = None
            print(f"[GROUND] load FAILED ({type(e).__name__}: {e}); "
                  f"callers fall back (cosine-only validity, no contradiction check).",
                  flush=True)
            return None, None


def classify(premise: str, hypothesis: str,
             model_name: str = _MODEL_DEFAULT, device: str = _DEVICE_DEFAULT):
    """3-class NLI for one (premise, hypothesis) pair.

    Returns {"entailment": p, "neutral": p, "contradiction": p} (floats summing to 1),
    or None if the model is unavailable (caller falls back). Premise is truncated to the
    pinned window; keep it SHORT (per-source memory, not the whole context) — short premise
    is the VRAM rule, not a latency one (activation spike grows O(seq^2))."""
    model, tok = get_model(model_name, device)
    if model is None:
        return None
    import torch
    enc = tok(premise, hypothesis, return_tensors="pt",
              truncation=True, max_length=_MAX_LEN)
    enc = {k: v.to(_dev) for k, v in enc.items()}
    with torch.no_grad():
        probs = model(**enc).logits.float().softmax(-1)[0].tolist()
    return {"entailment": probs[ENTAIL],
            "neutral": probs[NEUTRAL],
            "contradiction": probs[CONTRADICT]}


def contradiction_prob(evidence: str, claim: str, **kw):
    """P(claim CONTRADICTS evidence). evidence=premise, claim=hypothesis. None if unavailable.
    Write-time guard: evidence=existing stored memory, claim=new candidate fact."""
    r = classify(evidence, claim, **kw)
    return None if r is None else r["contradiction"]


def entailment_prob(evidence: str, claim: str, **kw):
    """P(evidence ENTAILS/supports claim). Read-time grounding: evidence=retrieved memory,
    claim=the answer. None if unavailable."""
    r = classify(evidence, claim, **kw)
    return None if r is None else r["entailment"]


def _prewarm_worker(model_name: str, device: str):
    """Load weights AND warm the inference path. The first real forward pass pays a one-time
    CUDA kernel-JIT / cuDNN-autotune cost on top of weight loading — measured live 2026-06-07
    at 458ms on the user's first memory turn, then ~30ms steady. Running one tiny dummy
    classify here moves that cost into the boot window (user is reading boot lines) instead of
    onto their first real turn. The dummy mirrors real usage (short memory-like premise + a
    user-like hypothesis) so it warms the same shapes. Best-effort: any failure just means the
    first real call warms it, exactly as before this polish."""
    model, _ = get_model(model_name, device)
    if model is None:
        return
    try:
        classify("The user likes tea.", "I prefer coffee.", model_name, device)
    except Exception:  # noqa: BLE001 — warmup is best-effort, never fatal
        pass


def prewarm(model_name: str = _MODEL_DEFAULT, device: str = _DEVICE_DEFAULT):
    """Load in a background daemon thread so the first real call doesn't pay the cold-load,
    and warm the inference path so it doesn't pay the first-forward CUDA warmup either.
    Idempotent: get_model returns the cached instance once loaded."""
    threading.Thread(
        target=_prewarm_worker, args=(model_name, device),
        name="grounding-prewarm", daemon=True,
    ).start()
