# Installing Vivianna — fully-local voice assistant (Windows)

_Last verified 2026-06-04 on the **only** machine it has ever run on: Windows 11, RTX 5070
12 GB, Ryzen 7 9700X, 32 GB RAM, Python 3.12.10. It has **not** been tested on other hardware,
other GPUs, AMD, or Linux. "Untested elsewhere" is honest; "doesn't run" is not — it does run,
here, end to end. If you're on a different box, expect to adapt the GPU/VRAM steps._

This guide is written to be followed **top to bottom with no prior knowledge**. If you copy the
exact folder layout in Step 1, **you will not have to edit a single line of code.**

---

## 0. What you're building

A 100% local assistant: speech-in (faster-whisper), brain (Qwen3.5-9B via llama.cpp),
speech-out (Kokoro), and local memory. The only optional network calls are web search
(Tavily/Exa) — everything else runs offline once installed.

**You need, on disk, four things:** (a) the llama.cpp runtime, (b) the model weights,
(c) a Python environment, (d) this `Vivianna` code folder. Steps below get each.

---

## 1. ⚠️ The directory layout — read this first, it's the #1 thing that breaks

**Every path in this project is hard-coded to the `F:\AI\` tree.** The launch scripts and
`config.py` use absolute paths. The foolproof path is to **reproduce this exact layout on a
drive `F:`** — then zero edits are needed:

```
F:\AI\
├── llama.cpp\                 ← the runtime (Step 3)  — contains llama-server.exe + DLLs
├── Models\                    ← the weights (Step 4)
│   ├── Qwen3.5-9B-UD-Q4_K_XL.gguf
│   ├── kokoro-v1.0.onnx
│   └── voices-v1.0.bin
├── asr-bench\
│   └── models\                ← faster-whisper cache (auto-fills on first /asr, or pre-copy)
├── API keys\                  ← optional: groq.txt / tavily.txt / exa.txt (Step 6)
└── Vivianna\                  ← THIS code folder
    ├── main.py, config.py, *.py, character_config.yaml, chat_template.jinja
    ├── requirements_full_2026-06-04.txt
    ├── run_vivianna.bat, start_server.bat
    ├── venv\                  ← you create this in Step 5
    └── data\                  ← memory + fastembed_cache (ships with the repo)
```

> **No drive `F:`?** You don't need a physical one. The paths are hard-coded to my dev layout —
> that's an artifact of where I built it, not a requirement of the software. Two ways to satisfy it:
>
> **Easiest (recommended — works on any machine, even a single-drive laptop): map a folder to `F:`.**
> Pick or make a folder on any drive with ~14 GB free, then in a terminal run:
> ```powershell
> mkdir C:\AI_root          # or any path/drive with space; skip if it exists
> subst F: C:\AI_root       # F:\ now points here — no code edits needed
> ```
> Now build the `F:\AI\...` tree from Step 1 normally (it lives inside `C:\AI_root`). **Caveat:**
> `subst` resets on reboot, so **re-run the `subst F: ...` line once each session before launching**
> (or drop it into a tiny `.bat` you run first / a logon task).
>
> **Permanent alternative (edit paths instead of mapping):** find-and-replace `F:\AI` → your base
> in **three files** — `config.py` (the `r"F:\AI\..."` lines: `_API_KEY_DIR`, `ASR_MODEL_DIR`,
> `MEMORY_DIR`) and **both** `run_vivianna.bat` / `start_server.bat` (the `llama-server.exe` path,
> the `-m` model path, the `--chat-template-file` path, and the `venv` / `cd` paths). Miss one and
> it half-starts — `subst` avoids that risk entirely, which is why it's the recommended route.
>
> _(Making the base path configurable so none of this is needed is a known TODO.)_

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
- **~14 GB free disk** (weights ~9 GB + venv incl. PyTorch ~3 GB + runtime ~0.7 GB).
- A terminal: **PowerShell** for the install commands below.

---

## 3. Get the llama.cpp runtime → `F:\AI\llama.cpp`

The brain runs on llama.cpp build **b9305, CUDA 13.1**. Either copy the `llama.cpp` folder from
the release bundle as-is, **or** fetch it:

1. Download `llama-b9305-cuda13.1.zip` **and** `cudart-cuda13.1.zip` from
   `https://github.com/ggml-org/llama.cpp/releases/tag/b9305`.
2. Extract **both** into `F:\AI\llama.cpp\` (the cudart zip provides `cudart64_*.dll` /
   `cublas64_*.dll` — without it the server won't start).
3. Confirm `F:\AI\llama.cpp\llama-server.exe` exists.

> Hashes for both zips are in `PROVENANCE.md` (verify if you fetched them yourself).
> Note: those NVIDIA `cudart`/`cublas` DLLs are under **NVIDIA's CUDA redistribution terms**,
> not MIT — fine to use, mind the terms if you re-distribute.

---

## 4. Get the model weights → `F:\AI\Models`

All four are free to download. Exact source repos, revisions, SHA256, and licenses are in
**`PROVENANCE.md`** — use it to verify bytes. Minimum required to run:

| File | Put at | Fetch from |
|---|---|---|
| `Qwen3.5-9B-UD-Q4_K_XL.gguf` (5.6 GB) | `F:\AI\Models\` | `huggingface.co/unsloth/Qwen3.5-9B-GGUF` → `Qwen3.5-9B-UD-Q4_K_XL.gguf` |
| `kokoro-v1.0.onnx` (310 MB) | `F:\AI\Models\` | `github.com/thewh1teagle/kokoro-onnx` release `model-files-v1.0` |
| `voices-v1.0.bin` (27 MB) | `F:\AI\Models\` | same kokoro-onnx release |

**Two smaller models download themselves on first run** (no action needed if you're online):
the memory embedder (`bge-small`, ships in `data\fastembed_cache`) and the ASR model
(`faster-whisper medium`, lands in `F:\AI\asr-bench\models` the first time you enable `/asr`).
The salience model (`deberta-v3-large`) also auto-pulls on demand and is **optional** — the
assistant runs fine if it never loads.

> **Optional / not required:** `Qwen3.5-9B-mmproj-BF16.gguf` (the vision projector). The launch
> scripts do **not** load it, so skip it unless you wire up image input yourself.

---

## 5. Create the Python environment

Open PowerShell **in `F:\AI\Vivianna`** and run:

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

This pulls ~170 pinned packages incl. faster-whisper, ctranslate2, kokoro/kokoro-onnx,
fastembed, transformers, and the `nvidia-*-cu12` runtime libs that ASR needs. Give it a few
minutes.

---

## 6. (Optional) Web-search API keys

Vivianna runs fully without these — they only enable live web lookups and the `/read` command.
To enable them, create plain-text files (key string only, nothing else):

- `F:\AI\API keys\tavily.txt` — Tavily key (general web; free tier at tavily.com)
- `F:\AI\API keys\exa.txt` — Exa key (semantic/research; free tier at exa.ai)
- `F:\AI\API keys\groq.txt` — **not needed** (legacy; ASR is local now). Leave absent.

If a file is missing, the app prints `[CONFIG] … key not found` and simply runs without that
feature. No crash.

---

## 7. (Recommended) Start from a clean memory

The shipped `data\` folder contains **dummy demo facts** (placeholder name/details used while
building). For your own use, clear them so Vivianna starts blank:

```powershell
Remove-Item F:\AI\Vivianna\data\memory_vectors.npy, `
            F:\AI\Vivianna\data\memory_meta.pkl, `
            F:\AI\Vivianna\data\chat_history.json -ErrorAction SilentlyContinue
```

Leave the `fastembed_cache\` and `jieba_cache\` subfolders alone — those are model caches, not
your data.

---

## 8. Run it

Double-click **`run_vivianna.bat`** (or run it from the terminal). It does everything in order:

1. launches `llama-server.exe` on `127.0.0.1:8080` with the tuned flags
   (`-c 8192 -ngl 99 --flash-attn on --jinja --reasoning off -ctk q5_1 -ctv q5_1`),
2. waits until the server reports healthy,
3. activates the venv, sets a UTF-8 console, and starts `python main.py`.

A second window titled **"Vivianna Server"** is the model server — leave it open. The first
launch is slower (the GPU compiles kernels and the model loads, ~10–20 s); after that it's fast.

**Type to chat.** Runtime toggles:

| Command | What it does |
|---|---|
| `/voice` | switch input from keyboard to **microphone** (loads + warms ASR on first use) |
| `/asr` | toggle the local speech-to-text engine on/off |
| `/tts` | toggle **spoken** replies (Kokoro) on/off — off by default, text-only |
| `/read <topic>` | fetch a page and read it aloud (needs a Tavily/Exa key) |

> `start_server.bat` starts **only** the model server (no Python app) — useful for debugging the
> server alone. For normal use, always use `run_vivianna.bat`.

---

## 9. Troubleshooting (the real failure modes)

| Symptom | Cause → Fix |
|---|---|
| Server window flashes and closes; app stuck "Waiting for server" | Missing **cudart/cublas DLLs** or wrong path. Re-extract `cudart-cuda13.1.zip` into `F:\AI\llama.cpp\`. Confirm `llama-server.exe` is there. |
| `… not found at F:\AI\…` / nothing loads | No `F:` drive (or `subst` not re-run after a reboot). Run `subst F: C:\AI_root` again, or use the find-replace alternative in Step 1. |
| `torch … could not find a version` / CPU-only torch | You skipped the **cu128 index** in Step 5. Reinstall torch with `--index-url https://download.pytorch.org/whl/cu128`. |
| Port 8080 already in use | Another process owns it. Change `--port` in **both** `.bat` files **and** `BASE_URL` in `config.py` to a free port. |
| `/asr` errors or no CUDA for ASR | The `nvidia-*-cu12` wheels didn't install, or driver too old. Re-run `pip install -r requirements…`; update the NVIDIA driver. As a fallback set `ASR_LOCAL_DEVICE = "cpu"` in `config.py` (slower, ~4 s floor). |
| Out of VRAM (8 GB card / other GPU resident) | Lower context `-c 8192`→`-c 4096`, or run **ASR on CPU** (`ASR_LOCAL_DEVICE="cpu"`) to free ~1.4 GB. TTS already runs on CPU (0 VRAM). |
| First Chinese reply stutters / logs spam | One-time `jieba` dictionary build (cached after). Expected; harmless. |
| It "remembers" facts you never said | That's the **dummy demo memory** — do Step 7. |
| Salience/`[SALIENCE]` never prints | Known: the salience judge is currently dormant. **Non-blocking** — the assistant works without it. |

---

## 10. Reset / kill switches

- **Stop everything:** close the Vivianna window and the "Vivianna Server" window.
- **Disable a cognitive layer:** set its `*_LAYER_ENABLED = False` in `config.py` (all are
  A/B-reversible flags).
- **Wipe memory:** repeat Step 7.

---

_Provenance, licenses, and byte hashes for every artifact: see `PROVENANCE.md`._
_Architecture, design doctrine, and the story of why it's built this way: see the README._
