# Installing Vivianna — fully-local voice assistant (Windows)

_Last verified 2026-06-09 on the **only** machine it has ever run on: Windows 11, RTX 5070
12 GB, Ryzen 7 9700X, 32 GB RAM, Python 3.12.10. It has **not** been tested on other hardware,
other GPUs, AMD, or Linux. "Untested elsewhere" is honest; "doesn't run" is not — it does run,
here, end to end. If you're on a different box, expect to adapt the GPU/VRAM steps._

This guide is written to be followed **top to bottom with no prior knowledge.**

---

## 0. What you're building

A 100% local assistant: speech-in (faster-whisper), brain (Qwen3.5-9B via llama.cpp),
speech-out (Kokoro), and local memory. The only optional network calls are web search
(Tavily/Exa) — everything else runs offline once installed.

**You need, on disk, four things:** (a) the llama.cpp runtime, (b) the model weights,
(c) a Python environment, (d) this `Vivianna` code folder. Steps below get each.

---

## 1. Where things go — paths are now configurable (no special drive needed)

Unzip/clone this `Vivianna` folder **anywhere** — any drive, any path, including a single-drive
laptop. Nothing is hard-coded to `F:\` anymore. There are two ways to tell Vivianna where the
big external files (the runtime + the LLM weights) live:

**Option A — default in-repo layout (zero config).** Drop the files into folders next to the
code and you don't have to set anything:

```
Vivianna\                         ← this code folder (put it anywhere)
├── main.py, config.py, *.py, character_config.yaml, chat_template.jinja
├── run_vivianna.bat, start_server.bat
├── requirements_full_2026-06-04.txt
├── .env.example                  ← template (only needed for Option B)
├── venv\                         ← you create this in Step 4
├── llama.cpp\                    ← the runtime (Step 2): llama-server.exe + its DLLs
├── models\                       ← the weights (Step 3)
│   ├── Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf   (what run_vivianna.bat expects)
│   ├── kokoro-v1.0.onnx
│   └── voices-v1.0.bin
├── keys\                         ← optional: groq.txt / tavily.txt / exa.txt (Step 5)
└── data\                         ← memory + caches (ships with the repo)
```

**Option B — files already live elsewhere? Point a `.env` at them.** Copy `.env.example` to
`.env` and fill in the paths (the file is gitignored, so your machine layout never ships):

```
LLAMA_EXE=D:\tools\llama.cpp\llama-server.exe
VIVIANNA_LLM_MODEL=D:\models\Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf
VIVIANNA_KEY_DIR=D:\secrets\vivianna-keys
# TTS/ASR model dirs and the data dir all have sensible <repo>-relative defaults — set
# them only if you keep those elsewhere too. See .env.example for the full list.
```

`.env` values win; anything you leave blank falls back to the default in-repo layout above.
**Either way, no source edits are required.**

---

## 2. Prerequisites

- **Windows 10/11**, 64-bit.
- **An NVIDIA GPU.** Reference build used 12 GB (RTX 5070). The full stack measures **~8.4 GB
  VRAM**, so 12 GB is comfortable, 8 GB is tight (see Troubleshooting → VRAM). Keep your
  **NVIDIA driver current** (the runtime is CUDA 13.x for llama.cpp / CUDA 12.x for ASR — a
  recent driver covers both; the CUDA *runtime DLLs ship with the binaries*, you do **not**
  install a CUDA toolkit).
- **Python 3.12** (built/verified on **3.12.10**). Install from python.org, tick *"Add Python
  to PATH"*. Other 3.1x may work but 3.12 is what the wheel pins target.
- **~15 GB free disk** (weights ~9 GB + venv incl. PyTorch ~3 GB + runtime ~0.7 GB + encoder
  models that auto-download to your user HF cache ~1.6 GB).
- A terminal: **PowerShell** for the install commands below.

---

## 3. Get the llama.cpp runtime

The brain runs on llama.cpp build **b9305, CUDA 13.1**. Either copy the `llama.cpp` folder from
the release bundle as-is, **or** fetch it:

1. Download `llama-b9305-cuda13.1.zip` **and** `cudart-cuda13.1.zip` from
   `https://github.com/ggml-org/llama.cpp/releases/tag/b9305`.
2. Extract **both** into one folder (the cudart zip provides `cudart64_*.dll` / `cublas64_*.dll`
   — without it the server won't start).
3. Place that folder at `Vivianna\llama.cpp\` (Option A), **or** set `LLAMA_EXE` in `.env` to the
   `llama-server.exe` inside it (Option B). Confirm `llama-server.exe` exists where you pointed.

> Hashes for both zips are in `PROVENANCE.md` (verify if you fetched them yourself).
> Note: the NVIDIA `cudart`/`cublas` DLLs are under **NVIDIA's CUDA redistribution terms**,
> not MIT — fine to use, mind the terms if you re-distribute.

---

## 4. Get the model weights

### 4a. Place by hand (the two large ones)

| File | Put at (Option A) | Fetch from |
|---|---|---|
| `Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf` (~5.6 GB) | `Vivianna\models\` | `huggingface.co/unsloth/Qwen3.5-9B-GGUF` (the **MTP** variant) |
| `kokoro-v1.0.onnx` (310 MB) | `Vivianna\models\` | `github.com/thewh1teagle/kokoro-onnx` release `model-files-v1.0` |
| `voices-v1.0.bin` (27 MB) | `Vivianna\models\` | same kokoro-onnx release |

> **Which Qwen GGUF?** `run_vivianna.bat` uses **MTP speculative decoding** (`--spec-type
> draft-mtp`) for ~1.38× faster output, so it expects the **MTP** variant
> (`Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf`). If you only have the standard `Qwen3.5-9B-UD-Q4_K_XL.gguf`,
> use **`start_server.bat`** instead (same model, no MTP) — or set `VIVIANNA_LLM_MODEL` to your
> file and remove the three `--spec-*` lines from `run_vivianna.bat`. Exact source repo,
> revision, SHA256 and license for the standard GGUF are in `PROVENANCE.md`.

### 4b. Auto-download (the four small ones — no action needed if you're online)

On first run these pull themselves into your user HuggingFace cache (`C:\Users\<you>\.cache\
huggingface\hub`) or the data dir, totalling ~1.6 GB:

- **memory embedder** `bge-small` → `data\fastembed_cache`
- **salience + rerank** `cross-encoder/ettin-reranker-150m-v1`
- **emotion** `SamLowe/roberta-base-go_emotions`
- **grounding NLI** `tasksource/deberta-small-long-nli`
- **ASR** `faster-whisper medium` → the ASR model dir, the first time you enable `/asr`

All four are Apache-2.0/MIT (see `PROVENANCE.md`). For an offline target, pre-stage them by
copying the cache folders or setting `HF_HOME` before first run.

---

## 5. Create the Python environment

Open PowerShell **in the `Vivianna` folder** and run:

```powershell
# 1. create + activate a 3.12 virtual environment
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
#   (if activation is blocked: Set-ExecutionPolicy -Scope Process RemoteSigned, then retry)

python -m pip install --upgrade pip

# 2. install GPU PyTorch FIRST from the CUDA 12.8 index (the plain PyPI build won't work)
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 3. install everything else (torch is already satisfied, so it won't be touched)
pip install -r requirements_full_2026-06-04.txt
```

This pulls ~174 pinned packages incl. faster-whisper, ctranslate2, kokoro/kokoro-onnx,
fastembed, transformers, **sentence-transformers** (for the ettin cross-encoder), and the
`nvidia-*-cu12` runtime libs that ASR needs. Give it a few minutes.

---

## 6. (Optional) Web-search API keys

Vivianna runs fully without these — they only enable live web lookups and the `/read` command.
To enable them, create plain-text files (key string only, nothing else) in `Vivianna\keys\`
(Option A) or in whatever folder you set as `VIVIANNA_KEY_DIR` (Option B):

- `tavily.txt` — Tavily key (general web; free tier at tavily.com)
- `exa.txt` — Exa key (semantic/research; free tier at exa.ai)
- `groq.txt` — **not needed** (legacy; ASR is local now). Leave absent.

If a file is missing, the app prints `[CONFIG] … key not found` and simply runs without that
feature. No crash. (`keys/` is gitignored — your keys never get committed.)

---

## 7. (Recommended) Start from a clean memory

The shipped `data\` folder may contain **dummy demo facts** (placeholder details used while
building). For your own use, clear them so Vivianna starts blank:

```powershell
Remove-Item .\data\memory_vectors.npy, .\data\memory_meta.pkl, .\data\chat_history.json -ErrorAction SilentlyContinue
```

Leave the `fastembed_cache\` and `jieba_cache\` subfolders alone — those are model caches, not
your data.

---

## 8. Run it

Double-click **`run_vivianna.bat`** (or run it from the terminal). It does everything in order:

1. loads `.env` (if present) and resolves paths,
2. launches `llama-server.exe` on `127.0.0.1:8080` with the tuned flags (incl. MTP speculative
   decoding for the speed-up),
3. waits until the server reports healthy,
4. activates the venv, sets a UTF-8 console, and starts `python main.py`.

A second window titled **"Vivianna Server"** is the model server — leave it open. The first
launch is slower (the GPU compiles kernels and the model loads, ~10–20 s); after that it's fast.

**Type to chat.** Runtime toggles:

| Command | What it does |
|---|---|
| `/voice` | switch input from keyboard to **microphone** (loads + warms ASR on first use) |
| `/asr` | toggle the local speech-to-text engine on/off |
| `/tts` | toggle **spoken** replies (Kokoro) on/off — off by default, text-only |
| `/read <topic>` | fetch a page and read it aloud (needs a Tavily/Exa key) |

> `start_server.bat` starts **only** the model server (the standard, non-MTP GGUF, no Python
> app) — useful for debugging the server alone. For normal use, always use `run_vivianna.bat`.

---

## 9. Troubleshooting (the real failure modes)

| Symptom | Cause → Fix |
|---|---|
| Server window flashes and closes; app stuck "Waiting for server" | Missing **cudart/cublas DLLs**, or `LLAMA_EXE` points nowhere. Re-extract `cudart-cuda13.1.zip` next to `llama-server.exe`; confirm the path (default `Vivianna\llama.cpp\`, or your `.env` `LLAMA_EXE`). |
| Server starts then exits with a model error | `VIVIANNA_LLM_MODEL` not found, or you pointed the **MTP** launcher at a **non-MTP** GGUF (or vice-versa). Check the file exists; match the launcher to the variant (Step 4a). |
| `… key not found at …keys` | Expected if you didn't add API keys — Vivianna runs without them. Add them per Step 6 to enable web search. |
| `torch … could not find a version` / CPU-only torch | You skipped the **cu128 index** in Step 5. Reinstall torch with `--index-url https://download.pytorch.org/whl/cu128`. |
| `ModuleNotFoundError: sentence_transformers` | Old requirements file. Re-run `pip install -r requirements_full_2026-06-04.txt` (it now pins `sentence-transformers==5.5.1`). Without it, salience + memory-rerank silently fall back. |
| Port 8080 already in use | Another process owns it. Change `--port` in **both** `.bat` files **and** `BASE_URL` in `config.py` to a free port. |
| `/asr` errors or no CUDA for ASR | The `nvidia-*-cu12` wheels didn't install, or driver too old. Re-run `pip install -r requirements…`; update the NVIDIA driver. As a fallback set `ASR_LOCAL_DEVICE = "cpu"` in `config.py` (slower, ~4 s floor). |
| Out of VRAM (8 GB card / other GPU resident) | Lower context `-c 8192`→`-c 4096`, or run **ASR on CPU** (`ASR_LOCAL_DEVICE="cpu"`) to free ~1.4 GB. TTS already runs on CPU (0 VRAM); the emotion model also runs on CPU. |
| First Chinese reply stutters / logs spam | One-time `jieba` dictionary build (cached after). Expected; harmless. |
| It "remembers" facts you never said | That's the **dummy demo memory** — do Step 7. |
| `[XENC] load FAILED` / no rerank | The ettin cross-encoder didn't load (usually missing `sentence-transformers` or no internet for first download). Non-fatal — memory falls back to cosine-only. Fix the dependency / go online once. |

---

## 10. Reset / kill switches

- **Stop everything:** close the Vivianna window and the "Vivianna Server" window.
- **Disable a cognitive layer:** set its `*_LAYER_ENABLED = False` in `config.py` (all are
  A/B-reversible flags).
- **Wipe memory:** repeat Step 7.

---

_Provenance, licenses, and byte hashes for every artifact: see `PROVENANCE.md`._
_Architecture, design doctrine, and the story of why it's built this way: see the README._
