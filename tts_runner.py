import time
import threading
import itertools
import queue
import numpy as np
import sounddevice as sd
from kokoro_onnx import Kokoro
from config import TTS_TAIL_DELAY, TTS_ENABLED, TTS_DEBUG, TTS_MODEL_PATH, TTS_VOICES_PATH, MEMORY_DIR

# Kokoro default-voice TTS (English-first). Replaces the Qwen3-TTS German
# voice-clone backend — no reference audio, no KikiriTest dependency. German +
# Chinese return later as language layers once the orchestration is proven in English.
MODEL_PATH    = TTS_MODEL_PATH    # fp32 — ~3.8x faster than int8 on this CPU (int8 dynamic-quant path is slower here)
VOICES_PATH   = TTS_VOICES_PATH
DEFAULT_SPEED = 1.0

# Mandarin G2P. Kokoro's espeak backend can't phonemize Chinese, so zh sentences are
# converted to Kokoro phonemes via misaki[zh] and fed in with is_phonemes=True.
# Loaded lazily — a pure-English session never imports misaki/jieba.
_zh_g2p = None


def _zh_phonemize(text: str) -> str:
    global _zh_g2p
    if _zh_g2p is None:
        import os
        import logging
        import jieba
        # jieba's prefix-dict cache defaults to %TEMP%, which Windows purges —
        # the same trap that broke the fastembed cache (recurring NO_SUCHFILE).
        # Pin it to the project data dir so the dict is built once and survives
        # temp wipes. Must be set before the first tokenize (i.e. before ZHG2P).
        _jieba_cache = os.path.join(MEMORY_DIR, "jieba_cache")
        os.makedirs(_jieba_cache, exist_ok=True)
        jieba.dt.tmp_dir = _jieba_cache
        # jieba logs its dict-build chatter at DEBUG straight to stderr, which
        # spliced into the live TTS response stream mid-sentence. Mute it.
        jieba.setLogLevel(logging.WARNING)
        from misaki import zh
        _zh_g2p = zh.ZHG2P()
    out = _zh_g2p(text)
    return out[0] if isinstance(out, tuple) else out

# Model + worker threads start lazily — only when TTS output is enabled, so a
# text-only session never pays the load/startup cost alongside llama.cpp.
_model           = None
_model_lock      = threading.Lock()
_workers_started = False

_synth_queue = queue.PriorityQueue()
_play_queue  = queue.Queue()
_seq         = itertools.count()

_lock              = threading.Lock()
_pending           = 0
_idle              = threading.Event()
_idle.set()

_tts_playing       = False
_tail_delay_active = False
_priority_active   = False

_enabled = False


def _load_model():
    global _model
    if _model is not None:
        return
    print("[TTS] Loading Kokoro (kokoro-v1.0 fp32)...", flush=True)
    _model = Kokoro(MODEL_PATH, VOICES_PATH)
    print("[TTS] Kokoro voice ready.", flush=True)


def _dec_pending() -> bool:
    global _pending
    with _lock:
        _pending -= 1
        return _pending == 0


def _synthesis_worker():
    global _priority_active
    while True:
        pri, _, payload = _synth_queue.get()
        if payload is None:
            _play_queue.put(None)
            break
        text, voice, lang = payload
        try:
            t0 = time.perf_counter()
            if lang == "zh":
                samples, sr = _model.create(_zh_phonemize(text), voice=voice,
                                            speed=DEFAULT_SPEED, is_phonemes=True)
            else:
                samples, sr = _model.create(text, voice=voice, speed=DEFAULT_SPEED, lang=lang)
            samples = np.asarray(samples, dtype=np.float32)
            if TTS_DEBUG:
                print(f"[TTS] synth done {time.perf_counter() - t0:.2f}s", flush=True)
            _play_queue.put((samples, sr, pri))
        except Exception as e:
            print(f"[TTS] synth error: {e}", flush=True)
            if _dec_pending():
                with _lock:
                    _priority_active = False
                _idle.set()


def _playback_worker():
    global _tts_playing, _tail_delay_active, _priority_active
    while True:
        item = _play_queue.get()
        if item is None:
            break
        samples, sr, pri = item
        try:
            with _lock:
                _tts_playing = True
            sd.play(samples, sr)
            sd.wait()
            if TTS_DEBUG:
                print(f"[TTS] play done {time.perf_counter():.2f}", flush=True)
        except Exception as e:
            print(f"[TTS] play error: {e}", flush=True)
        finally:
            with _lock:
                _tts_playing = False
            if _dec_pending():
                with _lock:
                    _tail_delay_active = True
                time.sleep(TTS_TAIL_DELAY)
                with _lock:
                    _tail_delay_active = False
                    _priority_active   = False
                _idle.set()


def _activate():
    """Load the model (once) and start the synth/playback threads (once)."""
    global _workers_started
    with _model_lock:
        _load_model()
        if not _workers_started:
            threading.Thread(target=_synthesis_worker, daemon=True).start()
            threading.Thread(target=_playback_worker,  daemon=True).start()
            _workers_started = True


def set_enabled(value: bool) -> bool:
    """Toggle spoken output. Enabling lazily loads the model on first use."""
    global _enabled
    value = bool(value)
    if value:
        _activate()
    _enabled = value
    print(f"[TTS] Output {'ON' if _enabled else 'OFF'}.", flush=True)
    return _enabled


def is_enabled() -> bool:
    return _enabled


def enqueue(text: str, voice: str, lang: str, priority: int):
    global _pending, _priority_active
    if not _enabled:
        return
    with _lock:
        _pending += 1
        if priority == 0:
            _priority_active = True
        _idle.clear()
    _synth_queue.put((priority, next(_seq), (text, voice, lang)))


def is_idle() -> bool:
    if not _enabled:
        return True
    with _lock:
        return (
            _pending == 0
            and not _tts_playing
            and not _tail_delay_active
            and not _priority_active
        )


def wait_idle(timeout: float = 30.0):
    if not _enabled:
        return
    _idle.wait(timeout=timeout)


# Honour the configured default at import time. When off, the model is NOT loaded.
if TTS_ENABLED:
    set_enabled(True)
else:
    print("[TTS] Output disabled (TTS_ENABLED=False) - text only. Type /tts to enable.",
          flush=True)
