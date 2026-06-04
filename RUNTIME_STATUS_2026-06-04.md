# Vivianna — Runtime Status

**Date:** 2026-06-04, 19:32 CEST (Europe/Berlin) (~3 weeks in) · **Snapshot source:** `tts_runner.py` (edited + verified live today), launch scripts + `F:\AI\Models\` (read live), prior `RUNTIME_STATUS_2026-06-03.md`.
**One-line:** Stack unchanged from 06-03 FULLY-LOCAL milestone. Today = **release-prep day**: verified the production model on disk, reviewed the demo clip, and fixed a latent jieba `%TEMP%` cache bug + log-spew in the Chinese TTS path. Supersedes `RUNTIME_STATUS_2026-06-03.md`.

---

## 1. Model & inference server  *(verified on disk today — unchanged)*
| Item | Value |
|---|---|
| Model | `Qwen3.5-9B-UD-Q4_K_XL` (`F:\AI\Models\Qwen3.5-9B-UD-Q4_K_XL.gguf`) — **stock Qwen, native alignment intact** |
| Server | llama.cpp `llama-server.exe` @ `127.0.0.1:8080` |
| Context | `-c 8192` · GPU offload `-ngl 99` · `--flash-attn on` |
| Template | `--jinja` + `chat_template.jinja` (think-fix), `--reasoning off` |
| KV cache | `-ctk q5_1 -ctv q5_1` |

**Verified 2026-06-04:** both live launch scripts (`run_vivianna.bat`, `start_server.bat`) load the stock gguf. `F:\AI\Models\` now holds only `Qwen3.5-9B-UD-Q4_K_XL.gguf` + `Qwen3.5-9B-mmproj-BF16.gguf` (vision projector — exclude from release if unused).

## 2. Generation parameters (`config.py`)  *(unchanged from 06-03)*
- `TEMPERATURE 0.5` · `MAX_TOKENS 2048` · `MAX_EXCHANGES 6` · `SHOW_REASONING False`
- `TOP_K 30` · `TOP_P 0.92` · `MIN_P 0.00`
- `REPEAT_PENALTY 1.01` · `FREQUENCY_PENALTY 0.1` · `PRESENCE_PENALTY 0.05` · `DRY_MULTIPLIER 0.05`

## 3. Input / Output
**ASR (input) — LOCAL:** faster-whisper / CTranslate2 4.7.2, model **`medium` / `cuda` / `int8`** (`asr.py`). Lazy-load + sm_120 warmup on enable. Toggles `/asr` (engine) and `/voice` (mic↔keyboard, auto-enables). `VOICE_INPUT False`, `ASR_ENABLED False` (lazy). `SAMPLE_RATE 16000`, `SILENCE 1.2s`, `VAD 3`, `POST_DELAY 1.5s`, `BEAM 5`, `LANGUAGE None` (auto EN↔DE). Models cached `F:\AI\asr-bench\models`. Groq cloud ASR removed (`GROQ_API_KEY` still loaded, unused).
**TTS (output):** `TTS_ENABLED False` (`/tts` toggles, lazy). **Kokoro v1.0 fp32 on CPU** → **0 GPU VRAM**. `SPEED 1.0`, `TAIL_DELAY 0.5s`. Per-sentence streaming emit + priority slot. EN + ZH (zh via `misaki[zh]`, voice `zf_xiaobei`); German deferred.
- **FIXED 2026-06-04 (`tts_runner.py` → `_zh_phonemize`):** jieba prefix-dict cache moved off volatile `%TEMP%` → **`F:\AI\Vivianna\data\jieba_cache`** (`jieba.dt.tmp_dir`, `os.makedirs`). Same Windows-purge trap as the fastembed fix. Also `jieba.setLogLevel(WARNING)` — its DEBUG dict-build chatter ("Building prefix dict…", "Loading model from cache…") was splicing into the live TTS response stream mid-sentence. Both verified in venv (no leak, cache relocated, tokenization correct). Lazy: takes effect on next ZH turn.

## 4. Memory  *(unchanged from 06-03)*
- Store: `data\memory_vectors.npy` + `memory_meta.pkl`; history `data\chat_history.json`.
- Embed `BAAI/bge-small-en-v1.5` (fastembed, **CPU** → 0 VRAM) · `TOP_K 5` · `PROACTIVE_MEMORY True` · `SUMMARY_TRIGGER` Jamie-tuned.
- Embed cache persistent at `F:\AI\Vivianna\data\fastembed_cache` (off `%TEMP%`).
- Compaction: overflow → running summary then trim. `[MEMORY] Compacted N msgs into summary` = summarization only (NOT the salience gate — see §5).
- **Note (release):** current `data\` memory entries are **dummy debug data** (fake name, blood type, organ line seen in the demo clip). Sanitize or swap for neutral demo facts before any public recording; state in README they are test data.

## 5. Cognitive Doctrine layers  *(unchanged from 06-03)*
| Layer | Flag | State |
|---|---|---|
| Emotion | ✅ | single-primary, decay |
| Salience | ✅ config | **⚠️ DORMANT — judge not firing (see open bug)** `STORE_THRESHOLD 0.35`, `RANK_WEIGHT 0.25`, `deberta-v3-large-zeroshot-v2.0`, device `auto` |
| Confidence | ✅ | from NLI certainty + memory grounding |
| Role / identity | ✅ | proceed/cautious/refuse |
| Ack (Phase A) | ✅ | `ACK_MODE="deterministic"`, `fire_ack 0.000s` |
`TIME_AWARENESS_ENABLED True` · `CHARACTER_SESSION_START_ONLY True` · all `DEBUG=True`.

## 6. Routing & tools (`router.py`)  *(unchanged from 06-03)*
- Deterministic guards before NLI: purge/remember, time-query → injected-time chat (never web), explicit source fetch.
- NLI bands: `WEB_CONFIDENCE_HIGH 0.85` · `MEDIUM 0.65` · `LOW 0.0`.
- `/read` explicit fetch-and-speak, `READ_MAX_CHARS 2500`. Keys in `F:\AI\API keys\`.

## 7. VRAM ledger (RTX 5070, 12 GB)  *(unchanged — measured 2026-06-03)*
- Full stack as running = **~8.35 GB used / ~3.6 GB free** (Qwen + ASR medium-int8 + TTS-on-CPU(0) + NLI). Repeatable.
- ASR medium/int8 delta ~1.37 GB; warm latency 0.27–0.85 s.
- DeBERTa salience projected +1.6 GB → ~9.95 GB / ~2.3 GB free. UNVERIFIED (judge dormant).

## 8. Today's changes (2026-06-04) — timestamped
- **18:19:08** — jieba cache + log-spew fix in `tts_runner.py` `_zh_phonemize` (see §3). Backup: `instant rollback\tts_runner_2026-06-04_181908.py`. Verified live in venv; temp test script removed.
- **~17:00–19:00** — Demo clip reviewed (see §9 release-prep). No code change from review beyond the jieba fix.

## 9. Open items / bugs
**Carried from 06-03 (still open — not touched today):**
1. **SALIENCE JUDGE DORMANT (priority):** DeBERTa auto-save not firing despite `SALIENCE_DEBUG=True` + 2× compaction, zero `[SALIENCE]` lines. Next: read `salience_layer.py` + invocation in `brain.py`.
2. **Secret-keeping gap:** confidentiality not enforced at memory layer — stored secret recited on recall; model confabulated "I corrected my internal record." Needs memory-level confidential flag + curb on false capability claims.
3. **DeBERTa → CPU** rebalance — moot until judge loads.
4. **Cleanup:** remove unused `GROQ_API_KEY` load · VAD 3 vs source-side gain/boom-arm tuning · German TTS · NLI band calibration.

**Release-prep (new, for the ~4-day window before Qwen Paw):**
5. **Sanitize demo memory set** (dummy name/blood-type/organ lines) and re-record the demo clip; cut a 30–45s establish→compact→recall arc (proves coherence, the actual design win).
6. **Do NOT bundle model weights** in the public repo — link to official Qwen + Kokoro downloads (the only real DMCA/license vector). Read Qwen + Kokoro licenses specifically.
7. **One clean-environment run** before promoting — honesty covers "untested on other hardware," not "doesn't run."
8. **README:** precise integration claim (no "first"/"only"/"none exists"), dummy-data note, verified-hardware line, free-API-key instructions, doctrine/story for the architecture rationale.
9. **Template = same repo with doctrine flags off**, not a hand-stripped parallel fork (avoid double-maintenance).
10. Narrow any "no hallucination recall" wording — stored "daily routine involves debugging" is a soft over-generalization from one session.

## 10. Kill switches / rollback
- Per-layer `*_LAYER_ENABLED=False` (A/B reversible). ASR: `/asr` off or `ASR_ENABLED`/device in config.
- File rollback (today): `instant rollback\tts_runner_2026-06-04_181908.py`. Prior: `224146`, `225316`, `230304` (06-03).
- Bench + tooling: `F:\AI\asr-bench\`.
