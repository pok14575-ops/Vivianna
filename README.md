[![Vivianna V0.1 Teaser](https://img.youtube.com/vi/uLzrKGEjsPg/maxresdefault.jpg)](https://youtu.be/uLzrKGEjsPg)

Will update for a V0.1 clip on a later date, this clip is just the old teaser for V0.1 , a placeholder

## What's Next

![V0.2 Teaser](Teaser%20v0.2.png)

*Preview of the upcoming V0.2 architecture and capability expansion.*

# Vivianna V0.1

> A fully local, Windows-first voice assistant / companion prototype built around local speech input, local LLM inference, local speech output, persistent memory, and modular cognitive routing layers.

Vivianna is an experimental local AI assistant project focused on one question:

**How much conversational continuity, memory persistence, emotional coherence, and practical household usefulness can be built on consumer hardware without depending on a cloud assistant platform?**

This is not a polished product, not a startup pitch, and not a claim of novelty. It is a working research / hobbyist release: transparent, messy where the prototype is still messy, and documented as honestly as possible.

V0.1 is the first release where the project is worth packaging as more than a work-in-progress dump: the runtime runs end-to-end on the reference machine, installation paths are configurable, provenance is documented, requirements are regenerated from the live environment, and the core local assistant loop is present.

---

## What Vivianna is

Vivianna is a local assistant architecture with:

- **Local LLM inference** through `llama.cpp`
- **Local speech-to-text** through `faster-whisper`
- **Local text-to-speech** through Kokoro ONNX
- **Persistent local memory** with semantic retrieval
- **Salience / rerank / grounding / emotion helper models**
- **Optional web search** through Tavily / Exa keys
- **A Windows launcher flow** that starts the model server and assistant together
- **Configurable paths** via repo-relative defaults or `.env`

The project is currently built for experimentation with local presence, voice interaction, memory, and assistant orchestration rather than benchmark chasing.

---

## What Vivianna is not

Vivianna V0.1 is **not**:

- a production assistant
- a medical, legal, safety, or emergency system
- a general autonomous agent framework
- a cloud replacement for frontier models
- tested across many machines
- guaranteed to install cleanly on arbitrary GPUs, Linux, macOS, AMD GPUs, or older CUDA setups

It runs on the reference system. Everything else should be treated as a porting/adaptation target.

---

## Current status

Validated on exactly one machine:

- **Windows 11**
- **AMD Ryzen 7 9700X**
- **NVIDIA RTX 5070 12 GB**
- **32 GB RAM**
- **Python 3.12.10**

The install guide is deliberately honest about this: V0.1 is known to run end-to-end there, but it has not been validated across other hardware.

---

## Runtime stack

| Layer | Current V0.1 component |
|---|---|
| Brain / LLM | `Qwen3.5-9B` GGUF through `llama.cpp` |
| Default LLM variant | `Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf` |
| Non-MTP fallback | `Qwen3.5-9B-UD-Q4_K_XL.gguf` |
| Server runtime | `llama.cpp` build `b9305`, CUDA 13.1 |
| ASR | `faster-whisper-medium`, CUDA INT8 |
| TTS | `kokoro-v1.0.onnx` fp32 + `voices-v1.0.bin` |
| Memory embedder | `qdrant/bge-small-en-v1.5-onnx-q` |
| Salience + memory rerank | `cross-encoder/ettin-reranker-150m-v1` |
| Emotion layer | `SamLowe/roberta-base-go_emotions` |
| Grounding / contradiction layer | `tasksource/deberta-small-long-nli` |
| Optional web | Tavily + Exa keys |
| Main platform | Windows + NVIDIA CUDA |

The default launcher uses the MTP / draft-decoding Qwen variant for faster output. The standard model path remains available for debugging or non-MTP use.

---

## What changed since the early WIP README

V0.1 is a large delta over the first public WIP snapshot:

- Paths are no longer hardcoded to a personal `F:\AI\...` layout.
- `.env` support allows model/runtime/key locations to live anywhere.
- The install flow now supports an in-repo default layout.
- The package inventory has been rebuilt from the live venv.
- The stale old `pip freeze` output is explicitly retired.
- `sentence-transformers` is now part of the requirements for the ettin cross-encoder path.
- The primary salience / rerank path moved to `ettin-reranker-150m-v1`.
- The retired DeBERTa-large zeroshot gate is no longer the default V0.1 path.
- The Qwen vision projector is not part of the V0.1 runtime path.
- The default launch path is the MTP Qwen GGUF, not the older standard-only path.
- Local ASR is the intended path; Groq is legacy and not required.
- The release now has dedicated `INSTALL.md`, `MANIFEST.md`, `PROVENANCE.md`, and pinned requirements.

---

## Repository contents

The repository contains the Vivianna source code, configuration, launch scripts, documentation, and requirements file.

It does **not** necessarily contain all model weights or external runtime binaries. Large artifacts are documented separately so users can fetch or place them correctly.

Important docs:

| File | Purpose |
|---|---|
| `README.md` | Project overview, status, architecture, limitations |
| `INSTALL.md` | Step-by-step install guide |
| `MANIFEST.md` | Dependency and artifact inventory |
| `PROVENANCE.md` | Source, license, revision, and SHA256 for external artifacts |
| `requirements_full_2026-06-04.txt` | Authoritative Python package list for V0.1 |
| `.env.example` | Optional path/key configuration template |

---

## Quick start

For the full install, follow `INSTALL.md` top to bottom.

The short version:

1. Clone or unzip the repo.
2. Install Python 3.12.
3. Create and activate a venv.
4. Install GPU PyTorch from the CUDA 12.8 PyTorch wheel index.
5. Install `requirements_full_2026-06-04.txt`.
6. Place `llama.cpp` build `b9305` and the model files either in the default repo-relative folders or point `.env` at them.
7. Optional: add Tavily / Exa API keys for web search.
8. Run `run_vivianna.bat`.

Minimal install command shape:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements_full_2026-06-04.txt
```

Do not use the old stale `pip freeze output.txt` from earlier development notes. V0.1 uses `requirements_full_2026-06-04.txt` as the authoritative environment file.

---

## Default folder layout

V0.1 can run with a repo-relative layout:

```text
Vivianna\
├── main.py
├── config.py
├── character_config.yaml
├── chat_template.jinja
├── run_vivianna.bat
├── start_server.bat
├── requirements_full_2026-06-04.txt
├── .env.example
├── llama.cpp\
│   └── llama-server.exe + required DLLs
├── models\
│   ├── Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf
│   ├── kokoro-v1.0.onnx
│   └── voices-v1.0.bin
├── keys\
│   ├── tavily.txt
│   └── exa.txt
└── data\
    ├── memory / cache files
    └── fastembed_cache\
```

Or you can keep the large files elsewhere and point `.env` at them:

```env
LLAMA_EXE=D:\tools\llama.cpp\llama-server.exe
VIVIANNA_LLM_MODEL=D:\models\Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf
VIVIANNA_KEY_DIR=D:\secrets\vivianna-keys
```

`.env` is intentionally gitignored so machine-specific paths and keys do not ship.

---

## Required external artifacts

The main manually placed artifacts are:

- `llama.cpp` CUDA runtime folder, build `b9305`
- `Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf`
- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

Additional smaller models are downloaded automatically on first use if internet is available:

- `qdrant/bge-small-en-v1.5-onnx-q`
- `cross-encoder/ettin-reranker-150m-v1`
- `SamLowe/roberta-base-go_emotions`
- `tasksource/deberta-small-long-nli`
- `Systran/faster-whisper-medium` when `/asr` is enabled

For exact URLs, revisions, SHA256 values, and licenses, read `PROVENANCE.md`.

---

## Runtime commands

Once Vivianna is running:

| Command | Effect |
|---|---|
| `/voice` | Switch from keyboard input to microphone input |
| `/asr` | Toggle local speech-to-text |
| `/tts` | Toggle spoken replies through Kokoro |
| `/read <topic>` | Fetch/read web content aloud, if web keys are configured |

TTS is off by default in the install guide flow. Use `/tts` when you want spoken replies.

---

## Locality and privacy model

Core operation is local once installed:

- LLM inference runs locally.
- ASR runs locally.
- TTS runs locally.
- Memory is stored locally.
- Helper classifiers run locally.

Optional network behavior:

- Tavily / Exa are only used if configured for web search / reading.
- Hugging Face downloads happen on first run unless models are pre-staged.
- No API keys should be committed; `keys/` and `.env` are expected to stay private.

---

## Memory warning

The shipped `data\` folder may contain dummy/demo memory from development. Before using Vivianna as your own assistant, reset memory:

```powershell
Remove-Item .\data\memory_vectors.npy, .\data\memory_meta.pkl, .\data\chat_history.json -ErrorAction SilentlyContinue
```

Do not delete model caches such as `fastembed_cache\` unless you want them to redownload.

---

## Known limitations

V0.1 is honest prototype software. Current limitations include:

- Validated on one Windows/NVIDIA machine only.
- 12 GB VRAM is the reference target; 8 GB may require reduced context or CPU ASR.
- Windows launch scripts are the maintained path.
- Linux, macOS, AMD GPUs, and non-CUDA setups are untested.
- Web retrieval requires optional third-party API keys.
- First-run model downloads require internet unless caches are pre-staged.
- Memory can be wrong and should not be treated as truth without review.
- The project is not a safety-critical system.
- Vision is not active in V0.1.
- The old Qwen3-TTS / cloned-voice route is not part of V0.1.
- The repository should not ship user API keys, personal memory, or local machine paths.

---

## Troubleshooting highlights

Common failure modes:

| Symptom | Likely cause |
|---|---|
| Server window closes immediately | Missing llama.cpp CUDA DLLs or wrong `LLAMA_EXE` |
| Server exits with model error | Wrong model path or MTP/non-MTP mismatch |
| Torch installs CPU-only | PyTorch CUDA wheel index was skipped |
| `sentence_transformers` missing | Old requirements file used |
| Port 8080 already in use | Another process is using the llama.cpp server port |
| ASR fails on CUDA | Missing CUDA wheels or old NVIDIA driver |
| It remembers fake facts | Demo memory was not cleared |

See `INSTALL.md` for the full troubleshooting table.

---

## Provenance and licenses

This repository's original source code, configuration, and documentation are licensed under the project license in `LICENSE`.

Third-party artifacts retain their own licenses. V0.1 documents every required model/runtime artifact in `PROVENANCE.md`, including source repository, revision, SHA256, and license.

High-level summary:

- Qwen GGUF weights: Apache-2.0
- Kokoro ONNX weights and voices: Apache-2.0
- bge-small embedder: Apache-2.0
- faster-whisper medium: MIT
- ettin reranker: Apache-2.0
- GoEmotions classifier: MIT
- long-NLI model: Apache-2.0
- llama.cpp runtime: MIT
- NVIDIA CUDA runtime DLLs: NVIDIA CUDA redistribution terms, not MIT

The NVIDIA CUDA DLLs are the main license split to be aware of when redistributing a bundled runtime.

---

## Philosophy

Vivianna is designed around perceptual coherence rather than raw benchmark maximalism.

The project prioritizes:

- local execution
- fast response initiation
- voice presence
- emotional readability
- persistent memory
- modular cognitive layers
- transparent failure modes
- household usefulness

The goal is not to pretend a 9B local model is a frontier cloud model. The goal is to make a small local system feel coherent, present, and useful by surrounding it with the right runtime, memory, speech, routing, grounding, and interaction design.

---

## Release note

V0.1 should be read as a real working snapshot, not a finished assistant.

It is the point where the project has crossed from “personal experiment on one machine” into “documented prototype someone else can inspect, learn from, and attempt to run.”

That distinction matters: the release is not claiming maturity. It is claiming enough structure, provenance, and installation honesty to be worth sharing.
