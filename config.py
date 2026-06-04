# =========================
# Vivianna Configuration
# =========================
import os
import yaml

_BASE = os.path.dirname(os.path.abspath(__file__))

# Load character config from YAML
with open(os.path.join(_BASE, "character_config.yaml"), "r", encoding="utf-8") as _f:
    _char = yaml.safe_load(_f)

# --- LLM ---
MODEL_NAME = "Qwen3.5-9B-UD-Q4_K_XL"
BASE_URL   = "http://127.0.0.1:8080/v1"
API_KEY    = "llama.cpp"

TEMPERATURE       = 0.5
MAX_EXCHANGES     = 6
MAX_TOKENS        = 2048
SHOW_REASONING    = False

TOP_K             = 30
TOP_P             = 0.92
MIN_P             = 0.00
REPEAT_PENALTY    = 1.01
FREQUENCY_PENALTY = 0.1
PRESENCE_PENALTY  = 0.05
DRY_MULTIPLIER    = 0.05

# --- API keys (all live in F:\AI\API keys\<name>.txt) ---
_API_KEY_DIR = r"F:\AI\API keys"

def _load_key(filename, label):
    path = os.path.join(_API_KEY_DIR, filename)
    try:
        with open(path, "r", encoding="utf-8") as _kf:
            return _kf.read().strip()
    except Exception:
        print(f"[CONFIG] {label} key not found at {path}.")
        return ""

GROQ_API_KEY   = _load_key("groq.txt",   "Groq (voice input)")
TAVILY_API_KEY = _load_key("tavily.txt", "Tavily (general web)")
EXA_API_KEY    = _load_key("exa.txt",    "Exa (semantic/research)")

VOICE_INPUT          = False   # True = mic ASR, False = keyboard (toggle at runtime with /voice)
ASR_SAMPLE_RATE      = 16000
ASR_SILENCE_DURATION = 1.2    # seconds of consecutive silence before stopping
ASR_VAD_MODE         = 3      # webrtcvad aggressiveness: 0 (lenient) – 3 (strict). 3 = reject ambient noise (noisy/nighttime room)
ASR_POST_DELAY       = 1.5    # seconds to wait after response before listening again

# --- Local ASR (faster-whisper / CTranslate2; replaces Groq cloud) ---
# medium/cuda/int8 chosen after benchmarking: best accuracy-per-VRAM on German + EN<->DE
# code-switch (5.6% WER on the hard clip; ~1.3 GB; 0.27-0.85 s warm). See F:\AI\asr-bench.
ASR_LOCAL_MODEL    = "medium"   # Whisper size; CT2 conversion auto-fetched if missing
ASR_LOCAL_DEVICE   = "cuda"     # "cuda" | "cpu"  (cpu has a ~4 s encoder floor — see bench)
ASR_LOCAL_COMPUTE  = "int8"     # int8 ≈ fp16 accuracy here at ~half the VRAM
ASR_LOCAL_LANGUAGE = None       # None = auto-detect (handles EN<->DE code-switch); or "de"/"en"
ASR_BEAM_SIZE      = 5
ASR_MODEL_DIR      = r"F:\AI\asr-bench\models"   # reuse already-downloaded CT2 models (no re-download)
ASR_ENABLED        = False      # load local engine at startup? False = lazy (toggle /asr); mirrors TTS_ENABLED

TTS_TAIL_DELAY       = 0.5    # seconds after sd.wait() before signaling idle (BT hardware buffer drain)
TTS_ENABLED          = False  # spoken output. False = text-only, model NOT loaded (toggle at runtime with /tts)
TTS_DEBUG            = False  # print [TTS] synth/play timing from worker threads. Off by default — those async
                             # prints race the input prompt and bleed onto the "You:" line. Errors always print.

# --- Temporal awareness (replaces the old keyword time/date tool) ---
TIME_AWARENESS_ENABLED = True  # inject real local date/time into every system prompt

# --- Web search routing (three-tier NLI confidence; consumed in router.py) ---
# Set against real [NLI-CALIB] data before trusting the bands — see calibration logging.
WEB_CONFIDENCE_HIGH   = 0.85   # >= HIGH   : silent auto-route to best source
WEB_CONFIDENCE_MEDIUM = 0.65   # >= MEDIUM : route + state the choice in one sentence
WEB_CONFIDENCE_LOW    = 0.0    # <  MEDIUM : ask once; re-route on the next turn (low-disruption)

# --- /read command (explicit fetch-and-speak; no NLI pass — see commands.parse_read_command) ---
READ_OPENING   = "Yes, I found it. Let me read it to you."  # spoken/printed before content
READ_CLOSING   = "That's everything."                       # spoken/printed after content
READ_MAX_CHARS = 2500   # cap on Tavily/Exa full-page text read aloud (Wikipedia uses its bounded summary extract)

# --- Memory ---
MEMORY_DIR          = r"F:\AI\Vivianna\data"
MEMORY_VECTORS_PATH = os.path.join(MEMORY_DIR, "memory_vectors.npy")
MEMORY_META_PATH    = os.path.join(MEMORY_DIR, "memory_meta.pkl")
HISTORY_PATH        = os.path.join(MEMORY_DIR, "chat_history.json")
MEMORY_TOP_K        = 5
MEMORY_EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
MEMORY_EMBED_CACHE  = os.path.join(MEMORY_DIR, "fastembed_cache")  # persistent cache; default %TEMP% gets purged by Windows -> model vanishes
PROACTIVE_MEMORY    = True
SUMMARY_TRIGGER     = 6

# --- Emotion Layer (Cognitive Doctrine V1) ---
EMOTION_LAYER_ENABLED = True   # False = identical to pre-layer behaviour (A/B testing)
EMOTION_DEBUG         = True   # print [EMOTION:*] state transitions

# --- Salience Layer (Cognitive Doctrine V1) ---
SALIENCE_LAYER_ENABLED  = True   # False = old binary auto-save (store whatever LLM approves)
SALIENCE_DEBUG          = True   # print [SALIENCE] scores
SALIENCE_STORE_THRESHOLD = 0.35  # min salience to auto-store an extracted fact
SALIENCE_RANK_WEIGHT    = 0.25    # retrieval re-rank weight; 0.0 = pure cosine (unchanged).
                                 # Set ~0.25 to bias retrieval toward high-salience memories.
# DeBERTa semantic salience scorer (PRIMARY gate; regex is fallback if unavailable).
SALIENCE_MODEL_ENABLED  = True    # False = regex-only (original V1 behavior)
SALIENCE_MODEL_NAME     = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
SALIENCE_MODEL_DEVICE   = "auto"  # "auto" -> cuda if available else cpu; or "cuda"/"cpu"

# --- Confidence Layer (Cognitive Doctrine V1) ---
CONFIDENCE_LAYER_ENABLED = True  # False = no confidence cue (direct wording always)
CONFIDENCE_DEBUG         = True  # print [CONFIDENCE] score/band

# --- Role / Identity-Preservation Layer (Cognitive Doctrine V1) ---
ROLE_LAYER_ENABLED = True   # False = never refuse/caution on role grounds
ROLE_DEBUG         = True   # print [ROLE] decisions (only when not 'proceed')

# --- Acknowledgement Coordinator (Tiered Cognition Doctrine, Phase A) ---
# Input-side ack buys conversational time + feeds a style promise into the prompt.
# Phase A uses the transparent DeterministicGate (regex, NO ML). One ack/turn.
ACK_LAYER_ENABLED = True    # False = no acks (identical to pre-ack behaviour, A/B)
ACK_DEBUG         = True    # print [ACK:*] resolution/emit trace
ACK_MODE          = "deterministic"  # "deterministic" (A) | "shadow" (B) | "live" (C)
ACK_NEUTRAL_FALLBACK_THRESHOLD = 0.75  # below this confidence -> neutral wording
                                       # (inert in Phase A: regex gate is always 1.0)

# --- Character (from YAML) ---
SYSTEM_PROMPT = _char.get("system_prompt", "").strip()

# Inject the full persona (SYSTEM_PROMPT) only on the FIRST turn of a session;
# regular turns run lean (base Qwen + cues + memory context + summary + history),
# with the stabilizer holding identity and the summary propagating it forward.
# True  = new behaviour (session-start-only persona; ~900 fewer tokens/turn).
# False = old behaviour (persona prepended to every turn). Flip to A/B live.
CHARACTER_SESSION_START_ONLY = True
