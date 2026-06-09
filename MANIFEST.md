# Vivianna — Dependency & Artifact Manifest

_Generated 2026-06-04. Inventory of everything the runtime depends on, including the
parts that live **outside** `F:\AI\Vivianna`. A scan scoped to the Vivianna folder
alone will miss the models, the llama-server binary, and the HuggingFace cache._

> ⚠️ **Source-of-truth note:** `F:\AI\pip freeze output.txt` is **STALE / not the live venv**.
> It lists 121 packages with version numbers that do not match the actual environment
> (e.g. torch 2.12.0 / transformers 5.9.0 vs. the real 2.11.0+cu128 / 4.57.3) and is
> **missing ~60 packages**, including the entire ASR stack. Use
> **`requirements_full_2026-06-04.txt`** (174 packages; regenerated from `venv` 2026-06-04,
> `sentence-transformers==5.5.1` added 2026-06-09 for the ettin cross-encoder) as the
> authoritative package list. The old txt should not be used to rebuild the venv.

---

## 1. Python environment — `F:\AI\Vivianna\venv`

- **174 packages** — authoritative list: `requirements_full_2026-06-04.txt`
- Key runtime pins:

| Role | Packages (live versions) |
|---|---|
| **LLM client** | `groq==1.2.0`, `openai==2.38.0`, `httpx`, `fastapi`, `uvicorn`, `pydantic` |
| **ASR (local)** | `faster-whisper==1.2.1`, `ctranslate2==4.7.2`, `av==17.0.1`, `webrtcvad-wheels==2.0.14`, `soundfile`, `librosa==0.11.0`, `soxr` |
| **TTS (Kokoro)** | `kokoro==0.9.4`, `kokoro-onnx==0.5.0`, `misaki==0.9.4`, `phonemizer-fork`, `espeakng-loader`, `sounddevice`, `num2words` |
| **TTS zh** | `jieba==0.42.1`, `pypinyin==0.55.0`, `pypinyin-dict==0.9.0`, `cn2an==0.5.24`, `proces` |
| **Encoder stack** (salience/rerank/emotion/grounding) | `sentence-transformers==5.5.1`, `transformers==4.57.3`, `torch==2.11.0+cu128`, `tokenizers`, `safetensors`, `accelerate==1.12.0`, `sentencepiece` |
| **Memory embedder** | `fastembed==0.8.0`, `onnxruntime==1.26.0`, `py_rust_stemmers`, `mmh3`, `numpy` |
| **GPU runtime (wheels)** | `nvidia-cublas-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cuda-nvrtc-cu12`, `nvidia-cudnn-cu12==9.23.0.39` |
| **Web fetch / read** | `ddgs`, `trafilatura`, `htmldate`, `courlan`, `jusText`, `lxml`, `primp`, `fake-useragent`, `requests` |
| **NLP / spaCy** | `spacy==3.8.14`, `thinc`, `blis`, `curated-transformers`, `spacy-curated-transformers` |
| **UI / misc** | `gradio==6.15.1`, `qwen-tts==0.1.1`, `pandas`, `scikit-learn`, `scipy`, `numba`, `loguru`, `rich`, `typer` |

> Note: both CPU-style torch and the `+cu128` GPU wheel chain are present; the live
> install is the **GPU** build (`torch==2.11.0+cu128`, `torchaudio==2.11.0+cu128`).

---

## 2. Native runtime — `F:\AI\llama.cpp`

- **`llama-server.exe`** — build **9305 (`63248fc3e`)**, CUDA 13.1, Clang 19.1.5, x86_64
- GPU libs: `ggml-cuda.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`, `cudart64_13.dll`,
  full `ggml-cpu-*.dll` per-microarch set, `libomp140.x86_64.dll`
- Build archives retained: `llama-b9305-cuda13.1.zip`, `cudart-cuda13.1.zip`

---

## 3. Models / weights

| Artifact | Location | Size | Runtime role |
|---|---|---|---|
| Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf | model dir (`VIVIANNA_LLM_MODEL`) | 5.71 GB | ✅ primary LLM — MTP variant (run_vivianna.bat, 1.38× decode) |
| Qwen3.5-9B-UD-Q4_K_XL.gguf | model dir (`VIVIANNA_LLM_MODEL_PLAIN`) | 5.56 GB | standard LLM (start_server.bat, no MTP) |
| kokoro-v1.0.onnx (fp32) | model dir (`VIVIANNA_TTS_MODEL`) | 311 MB | ✅ TTS (fp32 chosen, 3.8× over int8) |
| kokoro-v1.0.int8.onnx | model dir | 89 MB | superseded by fp32 |
| voices-v1.0.bin | model dir (`VIVIANNA_TTS_VOICES`) | 27 MB | ✅ Kokoro voice pack |
| bge-small-en-v1.5-onnx-q | `data\fastembed_cache` | 65 MB | ✅ memory embedder |
| cross-encoder/ettin-reranker-150m-v1 | HF cache (`C:\…\hub`) | ~569 MB | ✅ salience gate **+** memory rerank (one shared instance) |
| SamLowe/roberta-base-go_emotions | HF cache (`C:\…\hub`) | ~476 MB | ✅ emotion layer (28-label GoEmotions, CPU) |
| tasksource/deberta-small-long-nli | HF cache (`C:\…\hub`) | ~542 MB | ✅ grounding NLI (entail/contradict) |
| deberta-v3-large-zeroshot-v2.0 | HF cache (`C:\…\hub`) | 841 MB | optional fallback (`SALIENCE_BACKEND="zeroshot"`); ettin is the default |
| cross-encoder/ettin-reranker-17m-v1 | HF cache (`C:\…\hub`) | ~70 MB | test / fallback only |
| faster-whisper-medium (ct2) | ASR dir (`VIVIANNA_ASR_MODEL_DIR`) | 1.5 GB | ✅ ASR (chosen: medium/cuda/int8) |
| misaki g2p data | `venv\...\misaki\data` | small | ✅ TTS phonemization |

**Benchmark-only ASR models** (`F:\AI\asr-bench\models`, not in runtime path):
faster-whisper tiny / base / small / large-v3 / large-v3-turbo-int8, plus German
models (bofenghuang, mkenfenheuer, Zoont).

> ⚠️ **The encoder stack (ettin / go_emotions / long-NLI) + the whole HF cache live on `C:\`
> (default `~/.cache`), not in the model dir.** They auto-download on first run, so a clean
> install just needs internet once; to pre-bundle them for an offline machine, set `HF_HOME`
> before first run or copy the three `models--…` folders into the target's HF cache.

---

## 4. HF cache cruft — EXCLUDE from release

`C:\Users\49151\.cache\huggingface\hub` also contains models **not used by the current
runtime** (Kokoro replaced the Qwen3-TTS voice-clone route):

- `Qwen3-TTS-12Hz-*` (0.6B / 1.7B Base / CustomVoice / VoiceDesign + tokenizer) — old TTS route
- `grounding-dino-base`, `bert-base-uncased`, `drbaph--s2-pro-fp8`

---

## 5. User-supplied — NEVER ship

- `F:\AI\API keys\` — Groq, Tavily, Exa, etc. Excluded from release.

---

## 6. External ship list — binaries + weights (verified on disk 2026-06-04)

The lean, runtime-required set that lives **outside** `F:\AI\Vivianna`. Sizes are
ground-truth from disk, not estimates. This is the "undersell" list: only what the
runtime actually loads — benchmark/superseded/cruft is called out separately below.

### 6a. Native binaries — copy the whole `F:\AI\llama.cpp` folder

> ⚠️ **Correction:** the `.exe` files are **thin launchers** (`llama-server.exe` is
> 9.7 KB). The real payload is in the matching `-impl.dll`. Shipping the bare `.exe`
> will not run — you must ship the DLL chain alongside it.

| File | Size | Why required |
|---|---|---|
| `llama-server.exe` | 9.7 KB | launcher (entry point) |
| `llama-server-impl.dll` | 11.8 MB | ✅ actual server payload |
| `llama.dll` | 2.4 MB | ✅ core lib |
| `llama-common.dll` | 7.6 MB | ✅ shared lib |
| `mtmd.dll` | 1.2 MB | multimodal lib (ships with the folder; vision path unused in V0.1) |
| `ggml.dll`, `ggml-base.dll`, `ggml-rpc.dll` | ~0.9 MB | ✅ ggml core |
| `ggml-cuda.dll` | 148 MB | ✅ GPU backend |
| `ggml-cpu-*.dll` (per-microarch set) | ~16 MB total | ✅ CPU fallback (auto-selected at load) |
| `cublas64_13.dll`, `cublasLt64_13.dll`, `cudart64_13.dll` | ~508 MB | ✅ CUDA 13.1 runtime |
| `libomp140.x86_64.dll` | 0.6 MB | ✅ OpenMP |

- Build: **9305 (`63248fc3e`)**, CUDA 13.1, Clang 19.1.5, x86_64.
- Simplest correct action: **ship the entire `llama.cpp` folder** (the extra benchmark
  exes are tiny launchers; cost ≈ 0). Optionally drop the two build `.zip`s
  (`llama-b9305-cuda13.1.zip` 151 MB, `cudart-cuda13.1.zip` 384 MB) — they're archives,
  not runtime files.

### 6b. Model / weight files — required at runtime

| Artifact | Source location → ships to | Size | Role |
|---|---|---|---|
| `Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf` | model dir | 5.71 GB | ✅ primary LLM — MTP variant (run_vivianna.bat) |
| `Qwen3.5-9B-UD-Q4_K_XL.gguf` | model dir | 5.56 GB | standard LLM (start_server.bat) |
| `kokoro-v1.0.onnx` (fp32) | model dir | 311 MB | ✅ TTS |
| `voices-v1.0.bin` | model dir | 27 MB | ✅ Kokoro voice pack (stock `af_heart` EN + `zf_xiaobei` ZH) |
| `models--qdrant--bge-small-en-v1.5-onnx-q` | `data\fastembed_cache` | ~65 MB | ✅ memory embedder |
| `models--cross-encoder--ettin-reranker-150m-v1` | `C:\…\.cache\huggingface\hub` | ~569 MB | ✅ salience gate + memory rerank |
| `models--SamLowe--roberta-base-go_emotions` | `C:\…\.cache\huggingface\hub` | ~476 MB | ✅ emotion layer |
| `models--tasksource--deberta-small-long-nli` | `C:\…\.cache\huggingface\hub` | ~542 MB | ✅ grounding NLI |
| `models--Systran--faster-whisper-medium` | ASR dir | 1.46 GB | ✅ ASR (chosen: medium/cuda/int8) |

**Required-weights total ≈ 9.0 GB** (counting **one** LLM variant — MTP *or* standard, not both;
+ ~700 MB binaries ≈ **9.7 GB external footprint**). All four small encoder/embedder models
auto-download on first run — only the LLM + Kokoro files must be placed by hand.

> 📑 **Provenance captured** → see [`PROVENANCE.md`](PROVENANCE.md): per-artifact source
> repo+revision, SHA256, and license for all 8 weights + the llama.cpp runtime. Headline:
> **all 8 weights + llama.cpp are Apache-2.0 or MIT (redistributable).** Sole exception: the
> bundled NVIDIA CUDA runtime DLLs (`cudart`/`cublas`) are under **NVIDIA's CUDA redistribution
> terms, not open-source** — shippable, but honor those terms (or fetch-only).

> ⚠️ **The encoder stack (ettin / go_emotions / long-NLI) + the bge-small embedder are NOT in
> the model dir.** The three encoders live in the `C:\` HF cache; bge-small lives in the
> Vivianna data dir. All four **auto-download on first run**, so a fresh install just needs
> internet once. To pre-stage them offline, set `HF_HOME` before first run or copy the
> `models--…` folders into the target's HF cache.

### 6c. EXCLUDE — present on disk but NOT runtime-required

- **Superseded:** `kokoro-v1.0.int8.onnx` (88 MB — fp32 chosen instead).
- **Vision (out of V0.1):** `Qwen3.5-9B-mmproj-BF16.gguf` (879 MB) — the projector is not loaded
  (V0.1 uses the MTP draft-decode path; vision parked).
- **Benchmark-only ASR** (ASR dir, ~16 GB): all faster-whisper sizes except Systran-medium
  (tiny/base/small/large-v3/turbo-int8) + every German model (bofenghuang, mkenfenheuer,
  Zoont, `_hf_*_german`). None on the runtime path.
- **HF cache cruft** (`C:\…\hub`): `Qwen3-TTS-12Hz-*` (old voice-clone route, ~2.4 GB+),
  `ettin-reranker-17m-v1` (~70 MB, test/fallback only), `deberta-v3-large-zeroshot-v2.0`
  (841 MB — only if you flip `SALIENCE_BACKEND="zeroshot"`), `deberta-v3-base-zeroshot`
  (362 MB), `bert-base-uncased` (421 MB), `grounding-dino-base`, `drbaph--s2-pro-fp8`.
- **Build archives:** the two `.zip`s in `llama.cpp` (see 6a).

---

## Open items before this is release-grade

1. **Retire `F:\AI\pip freeze output.txt`** — it is not the live venv. Ship
   `requirements_full_2026-06-04.txt` instead.
2. **Relocate or explicitly bundle the `C:\`-side HF cache** (DeBERTa-large) so an
   `F:\AI`-scoped package doesn't silently drop the salience gate.
