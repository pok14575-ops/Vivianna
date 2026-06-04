import io
import os
import glob
import wave
import queue
import threading
import numpy as np
import sounddevice as sd
import webrtcvad
from config import (
    ASR_SAMPLE_RATE, ASR_SILENCE_DURATION, ASR_VAD_MODE,
    ASR_LOCAL_MODEL, ASR_LOCAL_DEVICE, ASR_LOCAL_COMPUTE,
    ASR_LOCAL_LANGUAGE, ASR_BEAM_SIZE, ASR_MODEL_DIR, ASR_ENABLED,
)

# Local ASR via faster-whisper (CTranslate2) — replaces the Groq cloud backend.
# medium/cuda/int8 (see config + F:\AI\asr-bench benchmarks). The model lazy-loads on
# enable so a text-only / TTS-only session never pays the ~1.3 GB VRAM + load cost,
# exactly like the TTS engine.

# CTranslate2 finds cuBLAS/cuDNN/cudart only if their pip-wheel bin dirs are on PATH
# BEFORE faster_whisper imports (os.add_dll_directory alone is not enough on Windows).
def _add_cuda_dlls():
    try:
        import nvidia  # namespace package -> __file__ is None, iterate __path__
        added = []
        for base in list(nvidia.__path__):
            for p in glob.glob(os.path.join(base, "*", "bin")):
                if os.path.isdir(p):
                    added.append(p)
        if added:
            os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + os.environ.get("PATH", "")
            for p in added:
                os.add_dll_directory(p)
    except Exception as e:
        print(f"[ASR] CUDA dll setup note: {e}", flush=True)

_add_cuda_dlls()
from faster_whisper import WhisperModel

_vad = webrtcvad.Vad(ASR_VAD_MODE)

# webrtcvad requires exactly 10 / 20 / 30 ms frames of 16-bit PCM
_FRAME_MS      = 30
_FRAME_SAMPLES = ASR_SAMPLE_RATE * _FRAME_MS // 1000   # 480 samples @ 16 kHz
_PREROLL       = 5                                       # frames to keep before speech starts (~150 ms)

# Lazy model + enable flag (mirrors tts_runner).
_model       = None
_model_lock  = threading.Lock()
_enabled     = False


def _load_model():
    global _model
    if _model is not None:
        return
    print(f"[ASR] Loading faster-whisper ({ASR_LOCAL_MODEL} / {ASR_LOCAL_DEVICE} / {ASR_LOCAL_COMPUTE})...", flush=True)
    _model = WhisperModel(
        ASR_LOCAL_MODEL,
        device=ASR_LOCAL_DEVICE,
        compute_type=ASR_LOCAL_COMPUTE,
        download_root=ASR_MODEL_DIR,
    )
    print("[ASR] Model ready.", flush=True)


def _warmup():
    # First CUDA call JITs sm_120 (Blackwell) kernels (~7 s). Do it on enable, not on the
    # user's first utterance, so live latency stays at the benchmarked ~0.3-0.9 s.
    try:
        silence = np.zeros(ASR_SAMPLE_RATE, dtype=np.float32)
        segs, _ = _model.transcribe(silence, beam_size=1, language=ASR_LOCAL_LANGUAGE)
        for _ in segs:
            pass
        print("[ASR] Warmup complete.", flush=True)
    except Exception as e:
        print(f"[ASR] Warmup note: {e}", flush=True)


def _activate():
    """Load the model (once) and warm it. Called on enable."""
    with _model_lock:
        _load_model()
        _warmup()


def set_enabled(value: bool) -> bool:
    """Toggle the local ASR engine. Enabling lazily loads + warms the model on first use."""
    global _enabled
    value = bool(value)
    if value:
        _activate()
    _enabled = value
    print(f"[ASR] Engine {'ON' if _enabled else 'OFF'}.", flush=True)
    return _enabled


def is_enabled() -> bool:
    return _enabled


def _to_pcm16(frame: np.ndarray) -> bytes:
    return (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _frames_to_wav(frames: list[bytes]) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(ASR_SAMPLE_RATE)
        for f in frames:
            wf.writeframes(f)
    return buf.getvalue()


def record_until_silence() -> bytes | None:
    silence_needed = int(ASR_SILENCE_DURATION * 1000 / _FRAME_MS)

    audio_q  = queue.Queue()
    speaking = False
    silence  = 0
    frames   = []
    pre_roll = []

    def _cb(indata, frame_count, time_info, status):
        audio_q.put(indata.copy())

    print("[ASR] Listening...", flush=True)

    try:
        with sd.InputStream(samplerate=ASR_SAMPLE_RATE, channels=1,
                            dtype="float32", blocksize=_FRAME_SAMPLES,
                            callback=_cb):
            while True:
                pcm      = _to_pcm16(audio_q.get().flatten())
                is_voice = _vad.is_speech(pcm, ASR_SAMPLE_RATE)

                if not speaking:
                    pre_roll.append(pcm)
                    if len(pre_roll) > _PREROLL:
                        pre_roll.pop(0)
                    if is_voice:
                        speaking = True
                        silence  = 0
                        frames.extend(pre_roll)
                        pre_roll.clear()
                        print("[ASR] Speech detected.", flush=True)
                else:
                    frames.append(pcm)
                    silence = 0 if is_voice else silence + 1

                    if silence >= silence_needed:
                        print("[ASR] Silence — stopping.", flush=True)
                        break

    except sd.PortAudioError as e:
        print(f"[ASR] Audio device error (headset disconnected?): {e}", flush=True)
        return None
    except Exception as e:
        print(f"[ASR] Recording error: {e}", flush=True)
        return None

    if not frames:
        return None

    print(f"[ASR] Recorded {len(frames) * _FRAME_MS / 1000:.1f}s", flush=True)
    return _frames_to_wav(frames)


def transcribe(wav_bytes: bytes) -> str:
    if not _enabled or _model is None:
        return ""
    segments, _info = _model.transcribe(
        io.BytesIO(wav_bytes),
        beam_size=ASR_BEAM_SIZE,
        language=ASR_LOCAL_LANGUAGE,
    )
    text = "".join(s.text for s in segments).strip()
    print(f"[ASR] → {text!r}", flush=True)
    return text


def listen_and_transcribe() -> str:
    if not _enabled:
        return ""   # engine off — caller (main loop) decides input mode
    wav = record_until_silence()
    if wav is None:
        return None   # device error signal (preserves main.py's failure counter)
    return transcribe(wav)


# Honour the configured default at import. When off, the model is NOT loaded.
if ASR_ENABLED:
    set_enabled(True)
else:
    print("[ASR] Engine disabled (ASR_ENABLED=False) — lazy. Type /asr or /voice to enable.",
          flush=True)
