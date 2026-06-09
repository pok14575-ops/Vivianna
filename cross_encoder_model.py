# cross_encoder_model.py
"""Shared, lazy, fallback-safe cross-encoder singleton.

ONE ettin-150m instance serves BOTH the salience gate (salience_layer.py) and
memory retrieval rerank (memory.py) — that shared 607MB instance is the whole
consolidation win over the old 1682MB DeBERTa-for-salience-only.

Thread-safe lazy load; any failure leaves callers on their own fallback (regex
salience / pure-cosine retrieval). Includes the tokenizer-class workaround for
ModernBERT reranker repos (e.g. Ettin) that declare a transformers-5.x-only
`TokenizersBackend` tokenizer class — on transformers 4.x that raises
"Unrecognized processing class", so we snapshot the repo, rewrite
tokenizer_config.json to a known fast-tokenizer class, and load from the copy.
The underlying model is plain ModernBertModel (supported on 4.x).
"""
from __future__ import annotations

import threading

# Off by default: the model is prewarmed in a daemon thread, so a success print
# would land on the live "You:" prompt and garble it (same reason as TTS_DEBUG).
try:
    from config import XENC_DEBUG as _XENC_DEBUG
except Exception:
    _XENC_DEBUG = False

_lock = threading.Lock()
_model = None            # the loaded CrossEncoder (singleton)
_failed = False          # set once load fails -> stay on fallback
_loaded_name = None


def get_model(model_name: str, device: str = "auto"):
    """Return the shared CrossEncoder, loading it on first call. None on failure.

    The first successful call fixes the model; later calls return that instance
    regardless of arguments (both consumers use the same configured model)."""
    global _model, _failed, _loaded_name
    if _model is not None:
        return _model
    if _failed:
        return None
    with _lock:
        if _model is not None:
            return _model
        if _failed:
            return None
        try:
            # Silence two benign load-time warnings that the prewarm thread would
            # otherwise dump onto the live "You:" input line, garbling the prompt:
            #   - transformers' `is_causal=False` notice (PROVEN benign: transformers
            #     5.10.2 gave identical eval numbers — see vivianna_reranker_bakeoff)
            #   - torch's `flop_counter`/triton "triton not found" warning
            import logging as _logging
            _logging.getLogger("torch.utils.flop_counter").setLevel(_logging.ERROR)
            # the is_causal notice is emitted by sentence_transformers' own logger
            # (base.modules.transformer.warn_if_unsupported), NOT transformers — so
            # target that module logger specifically.
            _logging.getLogger(
                "sentence_transformers.base.modules.transformer"
            ).setLevel(_logging.ERROR)

            import torch
            from sentence_transformers import CrossEncoder

            if device == "auto":
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                dev = device

            try:
                m = CrossEncoder(model_name, device=dev)
            except ValueError as e:
                if "processing class" not in str(e):
                    raise
                import json
                import os
                from huggingface_hub import snapshot_download
                local = snapshot_download(model_name)
                tcfg = os.path.join(local, "tokenizer_config.json")
                d = json.load(open(tcfg, encoding="utf-8"))
                d["tokenizer_class"] = "PreTrainedTokenizerFast"
                json.dump(d, open(tcfg, "w", encoding="utf-8"))
                m = CrossEncoder(local, device=dev)

            _model = m
            _loaded_name = model_name
            if _XENC_DEBUG:
                print(f"[XENC] loaded {model_name} on {dev}", flush=True)
            return _model
        except Exception as e:  # noqa: BLE001 — any failure -> fallback
            _failed = True
            _model = None
            print(f"[XENC] load FAILED ({type(e).__name__}: {e}); "
                  f"callers fall back.", flush=True)
            return None


def predict(model_name: str, device: str, pairs):
    """Convenience: lazy-load then score [[query, passage], ...]. None if unavailable."""
    m = get_model(model_name, device)
    if m is None:
        return None
    return m.predict(pairs, show_progress_bar=False)


def prewarm(model_name: str, device: str = "auto"):
    """Load the model in a background daemon thread so the first real call
    (salience gate / memory rerank) doesn't pay the ~6s cold-load. Idempotent:
    get_model returns the cached instance once loaded."""
    threading.Thread(
        target=get_model, args=(model_name, device),
        name="xenc-prewarm", daemon=True,
    ).start()
