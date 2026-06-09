# emotion_model.py
"""Shared, lazy, fallback-safe go_emotions classifier — the emotion layer's affective-tone
judge (encoder-stack-v2; see vivianna_encoder_stack_v2).

ONE `SamLowe/roberta-base-go_emotions` (125M) instance reads the USER message and returns a
per-label probability for all 28 GoEmotions categories. The emotion layer maps the relevant
ones (gratitude/love/caring -> warmth, curiosity/confusion -> curiosity, fear/nervousness ->
uncertainty) into candidates that compete in its existing single-primary contest, as a
SEMANTIC complement to its regex signals. Unmapped labels (incl. dominant `neutral` and the
near-dead grief/pride/relief) are simply ignored downstream.

Mirrors grounding.py / cross_encoder_model.py: thread-safe lazy load, prewarm daemon thread,
any failure leaves callers on their fallback (here: regex-only emotion).

multi_label model -> SIGMOID per label (independent), NOT softmax — several emotions can
co-fire (e.g. fear + nervousness), and probabilities do NOT sum to 1.

CPU by MEASURED decision (~15ms/pass on the Ryzen 9700X; the 28 categories all come out of one
forward pass, so the count is free). fp32 (fp16 CPU ops are unsupported/slow). GPU VRAM is
reserved for the LLM + the grounding NLI.
"""
from __future__ import annotations

import threading

try:
    from config import EMOTION_MODEL_NAME as _MODEL_DEFAULT
except Exception:
    _MODEL_DEFAULT = "SamLowe/roberta-base-go_emotions"
try:
    from config import EMOTION_MODEL_DEVICE as _DEVICE_DEFAULT
except Exception:
    _DEVICE_DEFAULT = "cpu"
try:
    from config import EMOTION_MODEL_MAX_LEN as _MAX_LEN
except Exception:
    _MAX_LEN = 256
try:
    from config import EMOTION_MODEL_DEBUG as _DEBUG
except Exception:
    _DEBUG = False

_lock = threading.Lock()
_model = None
_tok = None
_id2label = None
_failed = False
_dev = None


def get_model(model_name: str = _MODEL_DEFAULT, device: str = _DEVICE_DEFAULT):
    """Return (model, tokenizer), loading on first call. (None, None) on failure.
    First successful call fixes the instance; later calls return it regardless of args."""
    global _model, _tok, _id2label, _failed, _dev
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

            tok = AutoTokenizer.from_pretrained(model_name, model_max_length=_MAX_LEN)
            model = AutoModelForSequenceClassification.from_pretrained(model_name).to(dev).eval()

            _model, _tok, _dev = model, tok, dev
            _id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
            if _DEBUG:
                print(f"[EMO-MODEL] loaded {model_name} on {dev} "
                      f"({len(_id2label)} labels)", flush=True)
            return _model, _tok
        except Exception as e:  # noqa: BLE001 — any failure -> callers fall back (regex-only)
            _failed = True
            _model = _tok = None
            print(f"[EMO-MODEL] load FAILED ({type(e).__name__}: {e}); "
                  f"emotion falls back to regex-only.", flush=True)
            return None, None


def classify(text: str, model_name: str = _MODEL_DEFAULT, device: str = _DEVICE_DEFAULT):
    """Return {label: prob} for all 28 GoEmotions labels (independent sigmoid), or None if the
    model is unavailable (caller falls back to regex-only emotion). Labels are the model's own
    id2label (lowercased) so the mapping downstream is by NAME, not a brittle index."""
    if not text:
        return None
    model, tok = get_model(model_name, device)
    if model is None:
        return None
    import torch
    enc = tok(text, return_tensors="pt", truncation=True, max_length=_MAX_LEN)
    enc = {k: v.to(_dev) for k, v in enc.items()}
    with torch.no_grad():
        probs = model(**enc).logits.float().sigmoid()[0].tolist()
    return {_id2label[i]: probs[i] for i in range(len(probs))}


def _prewarm_worker(model_name: str, device: str):
    """Load weights AND warm the inference path (one tiny dummy classify) so the first real
    read doesn't pay the cold-load on the user's first turn. Best-effort: any failure just
    means the first real call warms it, exactly as before."""
    model, _ = get_model(model_name, device)
    if model is None:
        return
    try:
        classify("Thank you, that was really helpful.", model_name, device)
    except Exception:  # noqa: BLE001 — warmup is best-effort, never fatal
        pass


def prewarm(model_name: str = _MODEL_DEFAULT, device: str = _DEVICE_DEFAULT):
    """Load + warm in a background daemon thread so the first real read doesn't pay the
    cold-load. Idempotent: get_model returns the cached instance once loaded."""
    threading.Thread(
        target=_prewarm_worker, args=(model_name, device),
        name="emotion-model-prewarm", daemon=True,
    ).start()
