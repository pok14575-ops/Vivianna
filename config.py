# =========================
# Vivianna Configuration
# =========================
import os
import yaml

_BASE = os.path.dirname(os.path.abspath(__file__))


# --- Load machine-specific paths/keys from a .env (gitignored). ---
# Hand-rolled (no python-dotenv dependency). Lines are KEY=VALUE; blanks and
# lines starting with # are ignored. Real OS env vars take precedence over .env.
def _load_dotenv(path):
    try:
        with open(path, "r", encoding="utf-8") as _ef:
            for _line in _ef:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _, _v = _line.partition("=")
                _k = _k.strip()
                _v = _v.strip().strip('"').strip("'")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v
    except FileNotFoundError:
        pass


_load_dotenv(os.path.join(_BASE, ".env"))

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

# --- API keys (one <name>.txt per key in VIVIANNA_KEY_DIR; default <repo>/keys). ---
# Set VIVIANNA_KEY_DIR in .env to keep keys outside the repo (recommended).
_API_KEY_DIR = os.environ.get("VIVIANNA_KEY_DIR", os.path.join(_BASE, "keys"))

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
# code-switch (5.6% WER on the hard clip; ~1.3 GB; 0.27-0.85 s warm).
ASR_LOCAL_MODEL    = "medium"   # Whisper size; CT2 conversion auto-fetched if missing
ASR_LOCAL_DEVICE   = "cuda"     # "cuda" | "cpu"  (cpu has a ~4 s encoder floor — see bench)
ASR_LOCAL_COMPUTE  = "int8"     # int8 ≈ fp16 accuracy here at ~half the VRAM
ASR_LOCAL_LANGUAGE = None       # None = auto-detect (handles EN<->DE code-switch); or "de"/"en"
ASR_BEAM_SIZE      = 5
ASR_MODEL_DIR      = os.environ.get("VIVIANNA_ASR_MODEL_DIR", os.path.join(_BASE, "models", "asr"))  # CT2 models; auto-fetched here if missing
ASR_ENABLED        = False      # load local engine at startup? False = lazy (toggle /asr); mirrors TTS_ENABLED

# Kokoro model files (external download; default <repo>/models). Override in .env.
TTS_MODEL_PATH       = os.environ.get("VIVIANNA_TTS_MODEL",  os.path.join(_BASE, "models", "kokoro-v1.0.onnx"))
TTS_VOICES_PATH      = os.environ.get("VIVIANNA_TTS_VOICES", os.path.join(_BASE, "models", "voices-v1.0.bin"))
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

# --- Smart Search Router: two-stage source selection + empty-retrieval honesty (web_search.py) ---
# Both stages reuse the SAME prewarmed ettin-150m cross-encoder as salience / memory rerank
# (no extra VRAM, ~ms per handful of pairs). Fallback-safe at every stage: if the reranker
# is unavailable a stage just keeps the prior order, and if page fetches fail we fall back
# to the snippet — i.e. degrades to roughly the old behavior, never errors.
#   Stage 1 — rerank DDG snippets, keep the top-K candidates.
#   Stage 2 — fetch those K pages IN PARALLEL and re-rank on their ACTUAL extracted text, so
#             the model judges relevance on evidence, not a one-line preview. Also robust: if
#             the stage-1 #1 page has no extractable text, a lower one that does can still win.
# Fixes the observed "weather query -> Reddit opinion thread" misroute + confident
# confabulation (get_web_context used to take DDG's blind top-1 and never re-check content).
WEB_RERANK_ENABLED       = True
WEB_RESULTS_FETCH        = 8     # DDG snippet candidates fed to stage-1 (was a blind top-1)
WEB_FETCH_TOP_K          = 3     # top stage-1 results whose pages we fetch (parallel) for stage-2
WEB_FETCH_MIN_PAGES      = 2     # return once this many USABLE pages are in (+ the top candidate has
                                 # resolved) instead of waiting on stragglers — enough for a real
                                 # stage-2 content comparison. Observed live: 2/3 back fast, the 3rd
                                 # (a typosquat betting site) held the FULL timeout window for nothing.
WEB_FETCH_TIMEOUT        = 6.0   # hard wall-clock cap (s) — backstop for a genuine hang, incl. a slow
                                 # TOP-ranked page (which we DO wait for, vs skipping lower stragglers)
WEB_CONTENT_RERANK_CHARS = 2000  # chars of each fetched page fed to the stage-2 reranker (short = fast/low-VRAM)
# Relevance floor on the BEST *stage-2* (content) score. Below it, treat retrieval as empty
# and tell the user honestly rather than confabulate over an off-topic page. 0.0 = never drop
# (logging-only until calibrated against real [WEB-RANK2] scores — set once data exists,
# mirrors the WEB_CONFIDENCE_* "calibrate before trusting" note above).
WEB_RELEVANCE_FLOOR      = 0.0
WEB_RANK_DEBUG           = True  # print "[WEB-RANK]" (stage 1) + "[WEB-RANK2]" (stage 2) scores (calibration data)

# --- Item 1: web follow-up inheritance (vivianna_smart_search_router) ---
# After a web turn the fetched article is cached (runtime_state). A CHAT-routed follow-up that
# continues the topic is re-grounded over that cached article instead of free-associating (live
# 2026-06-08 turn 4: "speculations on locations?" routed to chat → invented wedding venues).
# Two trigger paths, by design:
#   • REFERENTIAL cue ("tell me more", "what about it") → acts immediately. Deterministic, needs
#     no threshold — a bare continuation right after a web turn unambiguously continues it.
#   • CONTENT follow-up (carries no referential marker, e.g. "speculations on locations?") → only
#     cross-encoder relatedness can detect it. Relatedness is ALWAYS logged ([WEB-FOLLOWUP]); the
#     floor below GATES this path and ships INERT (log-only) until calibrated from live data —
#     same discipline as WEB_RELEVANCE_FLOOR. 999.0 = nothing passes (the content path is dark;
#     referential cues still act). Lower it once real relatedness scores have accrued to switch
#     the turn-4 content-follow-up fix on. Re-grounding reuses the honesty prompt, so an off-topic
#     false-positive degrades to an honest "not in what I looked up", never a confident wrong answer.
WEB_FOLLOWUP_ENABLED           = True
WEB_FOLLOWUP_MAX_AGE_S         = 300    # cached web context older than this is stale → dropped
WEB_FOLLOWUP_RELATEDNESS_FLOOR = 999.0  # content-path gate; 999 = inert/log-only until calibrated
WEB_FOLLOWUP_DEBUG             = True   # print "[WEB-FOLLOWUP]" relatedness/decision lines

# --- Item 4: source-metadata transparency (publish date) ---
# Extract each fetched page's publish date (trafilatura/htmldate; None when undated — never
# guessed) and surface "(Source: <url>, published <date>)" + a staleness nudge in the grounded
# prompt, cache it, and answer provenance follow-ups ("what's your source / from what date")
# from that cached metadata instead of confabulating (live run 4 2026-06-08: she invented source
# dates, a rumor timeline, and named outlets she never accessed). Fallback-safe: any failure or
# a missing date degrades to no date / no claim of recency.
WEB_SOURCE_DATE_ENABLED = True

# --- /read command (explicit fetch-and-speak; no NLI pass — see commands.parse_read_command) ---
READ_OPENING   = "Yes, I found it. Let me read it to you."  # spoken/printed before content
READ_CLOSING   = "That's everything."                       # spoken/printed after content
READ_MAX_CHARS = 2500   # cap on Tavily/Exa full-page text read aloud (Wikipedia uses its bounded summary extract)

# --- Memory ---
MEMORY_DIR          = os.environ.get("VIVIANNA_DATA_DIR", os.path.join(_BASE, "data"))
MEMORY_VECTORS_PATH = os.path.join(MEMORY_DIR, "memory_vectors.npy")
MEMORY_META_PATH    = os.path.join(MEMORY_DIR, "memory_meta.pkl")
HISTORY_PATH        = os.path.join(MEMORY_DIR, "chat_history.json")
# Timestamped pre-edit / pre-audit memory backups. Default <repo>/backups;
# override with VIVIANNA_ROLLBACK_DIR to keep them off the project tree.
ROLLBACK_DIR        = os.environ.get("VIVIANNA_ROLLBACK_DIR", os.path.join(_BASE, "backups"))
MEMORY_TOP_K        = 5
MEMORY_EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
MEMORY_EMBED_CACHE  = os.path.join(MEMORY_DIR, "fastembed_cache")  # persistent cache; default %TEMP% gets purged by Windows -> model vanishes
PROACTIVE_MEMORY    = True
SUMMARY_TRIGGER     = 6
# Second-stage rerank: prefilter top-N by cosine(+salience), then reorder with the
# shared cross-encoder. Beats raw cosine on hard distractors (R@1 0.90->1.00 in
# test_reranker_compare --large); shares the salience model instance (no extra VRAM).
MEMORY_RERANK_ENABLED = True
MEMORY_RERANK_TOP_N   = 15   # cosine candidates fed to the cross-encoder per query

# --- Memory audit / dedup ---
# Store-time guard: when an AUTO-saved memory is a near-identical duplicate of an
# existing one (e.g. a recall turn re-extracting a fact already stored), skip the
# append instead of accumulating. High threshold => only near-exact matches; the
# softer semantic near-dups (cosine ~0.78 in practice) are left for /audit, which
# merges them with the LLM + user confirmation rather than lossy auto-skip.
MEMORY_DEDUP_GUARD       = True
MEMORY_DEDUP_THRESHOLD   = 0.95   # cosine >= this on an auto add() -> treat as dup, skip
# /audit: cosine >= this groups entries into a candidate cluster for LLM review.
# Calibrated to ~0.72 so semantic near-dups (observed 0.78) cluster while distinct
# facts (observed <=0.63) stay apart. The LLM is the final arbiter per cluster.
MEMORY_AUDIT_CLUSTER_COS = 0.72

# --- Emotion Layer (Cognitive Doctrine V1) ---
EMOTION_LAYER_ENABLED = True   # False = identical to pre-layer behaviour (A/B testing)
EMOTION_DEBUG         = True   # print [EMOTION:*] state transitions

# --- Emotion model: go_emotions affective-tone judge (encoder-stack-v2; vivianna_encoder_stack_v2) ---
# A SEMANTIC complement to the layer's regex signals: roberta-base-go_emotions (28-label
# multi_label sigmoid) reads the USER message and the 28 probs map -> warmth / curiosity /
# uncertainty candidates that compete in the layer's existing single-primary contest. Catches
# what regex misses ("I'm terrified about my daughter's op" -> uncertainty) without scripting.
# `focus` + `self_limit_discomfort` stay regex/architecture-driven; neutral (and dead facets
# grief/pride/relief) are never mapped, so dominant-neutral is suppressed by construction.
# CPU by MEASURED decision (~15ms/pass; GPU VRAM is reserved for the LLM + grounding NLI).
# Fallback-safe: model unavailable -> classify None -> pure regex behaviour. One input pass/turn,
# masked by the input ack + generation handshake. Off -> original V1 emotion (one flag flip).
EMOTION_MODEL_ENABLED   = True
EMOTION_MODEL_NAME      = "SamLowe/roberta-base-go_emotions"
EMOTION_MODEL_DEVICE    = "cpu"   # measured ~15ms/CPU pass; reserve GPU for LLM + grounding NLI
EMOTION_MODEL_MAX_LEN   = 256     # user turns are short; caps the CPU forward cost
EMOTION_MODEL_THRESHOLD = 0.30    # min sigmoid prob for a facet label to emit a candidate (tune live, 0.05–0.55)
EMOTION_MODEL_INTENSITY_SCALE = 1.0  # candidate intensity = prob * this (dial down if the model out-shouts regex)
EMOTION_MODEL_DEBUG     = False   # print "[EMO-MODEL] loaded ..." on load; load FAILURES always print

# --- Salience Layer (Cognitive Doctrine V1) ---
SALIENCE_LAYER_ENABLED  = True   # False = old binary auto-save (store whatever LLM approves)
SALIENCE_DEBUG          = True   # print [SALIENCE] scores
# THRESHOLD IS BACKEND-SPECIFIC. cross-encoder+softmax clusters scores high -> ~0.97.
# DeBERTa zero-shot wants ~0.35. If you flip SALIENCE_BACKEND, flip this too.
SALIENCE_STORE_THRESHOLD = 0.97  # cross-encoder(softmax) operating point; see test_reranker_compare --large
SALIENCE_RANK_WEIGHT    = 0.25    # retrieval re-rank weight; 0.0 = pure cosine (unchanged).
                                 # Set ~0.25 to bias retrieval toward high-salience memories.
# Semantic salience scorer (PRIMARY gate; regex is fallback if unavailable).
SALIENCE_MODEL_ENABLED  = True    # False = regex-only (original V1 behavior)
# Backend: "cross-encoder" = ettin-150m (DEPLOYED 2026-06-06; AUC 0.960 ~= DeBERTa 0.964
#   on the 64-item adversarial set, ~10x faster, 607MB vs 1682MB, doubles as the memory
#   reranker). "zeroshot" = the original DeBERTa-v3-large (set threshold back to ~0.35).
SALIENCE_BACKEND        = "cross-encoder"
SALIENCE_AGG            = "softmax"  # cross-encoder facet aggregation: facets compete (calibrated)
SALIENCE_AGG_TEMP       = 1.0
SALIENCE_MODEL_NAME     = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"  # used only when SALIENCE_BACKEND="zeroshot"
SALIENCE_MODEL_DEVICE   = "auto"  # "auto" -> cuda if available else cpu; or "cuda"/"cpu"
# Shared cross-encoder (salience gate + memory rerank). One instance, both roles.
CROSS_ENCODER_MODEL     = "cross-encoder/ettin-reranker-150m-v1"
CROSS_ENCODER_DEVICE    = "auto"
XENC_DEBUG              = False  # print "[XENC] loaded ..." on load. Off by default: prewarm loads it
                                 # async, so the success line would land on the live "You:" prompt.
                                 # Load FAILURES always print regardless.

# --- Grounding / contradiction NLI (encoder-stack-v2 "third piece"; see grounding.py) ---
# ONE deberta-small-long-nli (142M) answers POLARITY (entail/neutral/contradict) — the one
# question the ettin reranker structurally CANNOT (it scores topical relevance, blind to
# agree-vs-conflict). Two consumers, wired in grounding steps 3c/3d: read-time grounding
# (stabilizer.pre_generate, augments cosine-only memory_valid) + write-time contradiction
# guard (_auto_save_memory). fp16 on CUDA is PARITY-PROVEN vs fp32 (0 decision flips,
# _encoder_fp16_fp32_bench.py; ~301MB peak). Live co-residence eyeball after prewarm is wired.
GROUNDING_ENABLED       = True   # False = cosine-only validity, no contradiction guard (gates prewarm too)
GROUNDING_MODEL         = "tasksource/deberta-small-long-nli"
GROUNDING_DEVICE        = "auto" # "auto" -> cuda(fp16) if available else cpu(fp32)
GROUNDING_MAX_LEN       = 1680   # tokenizer pinned to model max_position_embeddings (label order e/n/c verified)
GROUNDING_DEBUG         = False  # print "[GROUND] loaded ..." on load; load FAILURES always print
# Read-time grounding (3c, Option A): mark a retrieved memory conflict=True when
# NLI P(memory CONTRADICTS the user's current message) >= this. Smoke showed conflict ~0.94
# vs agreement ~0.14 (wide margin); conservative default — a false conflict blocks that turn's
# auto-save + fires the "memory uncertain" cue. Tune via the [STABILIZER:GROUND] per-turn log.
GROUNDING_CONTRADICT_THRESHOLD = 0.60

# --- Web-answer grounding guard (answer-precision; see brain._web_answer_grounding) ---
# After a web-grounded answer is generated, score each FACTUAL sentence for entailment by the
# fetched page (NLI premise=page text, hypothesis=sentence) and report the weakest-supported
# one — the most likely confabulation. Source selection (the two-stage rerank) picks the right
# page; THIS guards the answer the model writes over it. Motivated live 2026-06-08: the router
# picked the correct wedding article, but the 9B fabricated a specific date + venue the page
# never contained — caught by a domain-expert listener, not the system. Reuses the SAME
# grounding NLI (no extra VRAM); per-sentence passes are sequential (same peak VRAM as 3c/3d).
# LOG-ONLY: never suppresses or hedges yet — gathers [WEB-GROUND] entailment data to calibrate
# the floor first (same discipline as WEB_RELEVANCE_FLOOR; honours the memory-governance
# "detect autonomously, don't mutate without authority" line — here we only detect + log).
WEB_GROUNDING_CHECK_ENABLED  = True   # also gated on GROUNDING_ENABLED; False = no check at all
WEB_GROUNDING_ENTAIL_FLOOR   = 0.50   # tag a sentence "LOW" when entailment < this (LOG TAG ONLY — no action yet)
WEB_GROUNDING_MIN_SENT_CHARS = 25     # skip shorter sentences + questions (not factual claims worth grounding)
# 5a — disclaimer/meta filter: an HONEST sentence that says the answer ISN'T in the page ("I
# could not find X", "the search results do not contain Y", "I would need to check USDA") is
# genuinely NOT entailed by the page, so the entailment guard above flags it LIKELY-CONFABULATION
# — exactly backwards (live 2026-06-08 turn 6, the session's most honest answer, scored 0.092).
# Entailment cannot separate confab from honest disclaimer; a sentence matching one of these
# narrow markers is skipped (not scored), so only true world-claims reach the guard. Markers are
# deliberately TIGHT (specific first-person inability/intent + references to the retrieval
# itself) to avoid swallowing a real confab that merely opens with "I". Substring match, lowered.
WEB_GROUNDING_DISCLAIMER_MARKERS = (
    # first-person inability / intent — meta about what Vivianna can/can't do, not a world-claim
    "i cannot", "i can't", "i couldn't", "i could not", "i was unable", "i wasn't able",
    "i was not able", "i am unable", "i'm unable", "i would need", "i'd need",
    "i do not have", "i don't have", "i don't see", "i do not see",
    # references to the retrieval / source material itself
    "the search results", "the search did not", "the search didn't", "the provided",
    "the article does not", "the article doesn't", "the text does not", "the text doesn't",
    "the page does not", "the page doesn't", "the source does not", "the source doesn't",
    "does not contain", "doesn't contain", "no information", "no relevant",
    "could not find any", "couldn't find any", "didn't return", "did not return",
)

# --- Time-gated contradiction clarify-and-resolve (extends 3c; vivianna_contradiction_clarify_plan) ---
# When a retrieved memory contradicts the user's current message (the 3c path), AND that
# memory is OLD (effective age >= CLARIFY_MIN_AGE_HOURS), Vivianna ASKS at the end of her reply
# whether it changed, then resolves on the next turn: update (replacement) / delete (bare
# negation) / keep+re-stamp (re-confirmation). A contradiction vs a FRESH memory is treated as
# transient noise (mood/joke) and only the existing 3c honesty cue fires (no question). Age uses
# confirmed_at if present else created_at; a memory with NO timestamp (legacy) counts as old ->
# eligible. Flag-gated; off -> pure 3c behavior (one flip reverts).
CLARIFY_ENABLED        = True
CLARIFY_MIN_AGE_HOURS  = 12.0   # contradicted-memory age below this = noise, no clarify question

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
