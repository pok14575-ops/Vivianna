# Artifact Provenance — Vivianna V0.1 release

_Originally generated 2026-06-04; updated 2026-06-09 for V0.1 (the encoder-stack swap:
ettin reranker + go_emotions + long-NLI added; the Qwen vision projector and the
DeBERTa-large zeroshot gate retired). Companion to MANIFEST §6 (file inventory). This file
answers the three questions a copier actually needs: **where each artifact came from, that
they got the right bytes (SHA256), and whether redistribution is allowed.**_

## Headline finding
**All 9 model weights + the llama.cpp runtime are Apache-2.0 or MIT — legally redistributable
as blobs.** (#1 and #1b are two variants — standard and MTP — of the same Qwen3.5-9B LLM.) No weight is license-restricted, so nothing is *forced* to be fetch-only;
`fetch-only` below is a size/convenience recommendation only, never a legal requirement.
**The one exception is the NVIDIA CUDA runtime (#10):** its `cudart`/`cublas` DLLs are *not*
open-source — they are redistributable, but only under **NVIDIA's CUDA redistribution terms**
(the CUDA Toolkit EULA's redistributable-components clause). Legally shippable with attribution
to those terms; simplest clean path is **fetch-only** (let the consumer pull NVIDIA's own
`cudart` redistributable, or take it via the llama.cpp CUDA release). There is no cloned-voice
path (`victoria.npy` deleted 2026-06-04), so there is no voice-rights exposure either.

## How to read this
- **SHA256** is over the exact file at the listed path. For the five HuggingFace-cache
  artifacts (bge, faster-whisper, ettin, go_emotions, long-NLI) the hash is of the **primary
  weight file** and — because HF stores LFS objects under their own SHA256 — that hash equals
  the on-disk blob name, i.e. it is self-verifying against HF. The exact-revision anchor for
  those is the **pinned commit hash** in the source URL (re-pulling that commit yields
  bit-identical files).
- All hashes are lowercase SHA256.

## Provenance table

| # | Artifact (local name) | Source (repo @ revision) | License | SHA256 (primary file) | Ship-as |
|---|---|---|---|---|---|
| 1 | `Qwen3.5-9B-UD-Q4_K_XL.gguf` (5.56 GiB) — standard LLM (start_server.bat) | `huggingface.co/unsloth/Qwen3.5-9B-GGUF` @ `main` (file `Qwen3.5-9B-UD-Q4_K_XL.gguf`) | Apache-2.0 | `6f5d30666c2d8ae16a306e616d95341dcf3cc46810df84d7e6f5a7d1e4c1b293` | blob (or fetch) |
| 1b | `Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf` (5.71 GiB) — **MTP draft variant** (run_vivianna.bat default; 1.38× decode) | `huggingface.co/unsloth/Qwen3.5-9B-GGUF` (MTP/multi-token-prediction variant of Qwen3.5-9B) | Apache-2.0 | `362f85a2d7dbc0259e926d5ac33ca0d0f17fd3753496d65bfd2106384c929d3f` | blob (or fetch) |
| 2 | `kokoro-v1.0.onnx` fp32 (310 MiB) | `github.com/thewh1teagle/kokoro-onnx` release `model-files-v1.0` | Apache-2.0 (weights) | `7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5` | blob |
| 3 | `voices-v1.0.bin` (26.9 MiB) | `github.com/thewh1teagle/kokoro-onnx` release `model-files-v1.0` | Apache-2.0 (weights) | `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d` | blob |
| 4 | `models--qdrant--bge-small-en-v1.5-onnx-q` (folder ~65 MB; weight `model_optimized.onnx` 63.4 MiB) | `huggingface.co/qdrant/bge-small-en-v1.5-onnx-q` @ `52398278842ec682c6f32300af41344b1c0b0bb2` | Apache-2.0 | `51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431` | blob (or fetch) |
| 5 | `models--Systran--faster-whisper-medium` (folder 1.46 GB; weight `model.bin` 1.42 GiB) | `huggingface.co/Systran/faster-whisper-medium` @ `08e178d48790749d25932bbc082711ddcfdfbc4f` | MIT | `9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae` | blob (or fetch) |
| 6 | `models--cross-encoder--ettin-reranker-150m-v1` (weight `model.safetensors` ~569 MiB) | `huggingface.co/cross-encoder/ettin-reranker-150m-v1` @ `025501c4e0f9bbeb4c5b198318e0089ff061cc14` | Apache-2.0 | `0dd04496917f100307bc37e0fc7acb8750bb8781badfdb21a368b0e965a0d981` | fetch (on `C:\`) |
| 7 | `models--SamLowe--roberta-base-go_emotions` (weight `model.safetensors` ~476 MiB) | `huggingface.co/SamLowe/roberta-base-go_emotions` @ `d75048347613a25d77de8cf6412eaae9fa7b26be` | MIT | `84d6d338b4cf63f0ed3c990a0ce748d32d1d2965c072f4645accaa71af3888c0` | fetch (on `C:\`) |
| 8 | `models--tasksource--deberta-small-long-nli` (weight `model.safetensors` ~542 MiB) | `huggingface.co/tasksource/deberta-small-long-nli` @ `9a77395d4d3751be9e2a69c4ae318491d9b3fffb` | Apache-2.0 | `9af30c7ad7235a2054300bc2df1d98149ad6008dd1ef06212be8b32b5d1b3458` | fetch (on `C:\`) |
| 9 | `llama.cpp` runtime — build **9305** (`63248fc3e`), CUDA 13.1 | `github.com/ggml-org/llama.cpp` release `b9305`; retained asset `llama-b9305-cuda13.1.zip` | MIT | `882df5e511259e4976c137b06ff56a2c0aa6433811b350a26e7e828aea5beea6` (zip) | fetch (whole folder, see §9a) |
| 10 | CUDA 13.1 runtime DLLs | NVIDIA CUDA runtime, bundled with the llama.cpp CUDA release; retained asset `cudart-cuda13.1.zip` | **NVIDIA CUDA redistribution terms** (NOT open-source — see note) | `f96935e7e385e3b2d0189239077c10fe8fd7e95690fea4afec455b1b6c7e3f18` (zip) | fetch (with #9) |

## Per-artifact fetch instructions / notes

### 1 / 1b. Qwen3.5-9B GGUF — `unsloth/Qwen3.5-9B-GGUF`
- Source confirmed from the GGUF's own embedded metadata, not guessed:
  `general.quantized_by = Unsloth`, `quantize.imatrix.file = Qwen3.5-9B-GGUF/imatrix_unsloth.gguf`,
  `general.license = apache-2.0`, base model `Qwen/Qwen3.5-9B`. Repo license on HF = `apache-2.0`
  (re-confirmed 2026-06-09; the repo documents the MTP / multi-token-prediction variant).
- Fetch (standard): `https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf`
- **1b — MTP variant** (`Qwen3.5-9B-MTP-UD-Q4_K_XL.gguf`): the default model for `run_vivianna.bat`,
  enabling MTP speculative decoding (`--spec-type draft-mtp`, ~1.38× decode). Same Apache-2.0
  Qwen3.5-9B lineage as #1. `start_server.bat` uses the standard #1 instead.
- Revision: both live under `Models/` as plain files (not HF-cache layout), so there is no
  cached commit pin. The **SHA256s above are the byte anchors** — verify after pull. Upstream
  `main` may advance; if hashes differ, the repo was re-uploaded.
- **Note:** V0.1 ships **no vision projector** (`mmproj-BF16.gguf`). The MTP draft-decode path
  (the 1.38× speedup) is used instead, and the launch scripts do not load a projector.

### 2–3. Kokoro TTS — `thewh1teagle/kokoro-onnx`
- Model weights are Apache-2.0 (derived from `hexgrad/Kokoro-82M`); the kokoro-onnx project
  code is MIT. For these two **weight** blobs the governing license is Apache-2.0.
- Fetch (release tag pins the revision):
  - `https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx`
  - `https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin`
- Runtime uses stock voices `af_heart` (EN) + `zf_xiaobei` (ZH) from `voices-v1.0.bin`.

### 4. bge-small embedder — `qdrant/bge-small-en-v1.5-onnx-q`
- Apache-2.0 ONNX/quantized port of `BAAI/bge-small-en-v1.5`. Lives in the Vivianna data dir
  (`data/fastembed_cache`), **not** the model dir. Auto-downloads on first run.
- Re-pull pinned commit: `huggingface.co/qdrant/bge-small-en-v1.5-onnx-q` @
  `52398278842ec682c6f32300af41344b1c0b0bb2`.

### 5. faster-whisper ASR — `Systran/faster-whisper-medium`
- MIT. CTranslate2 model; runtime config = medium / cuda / int8. Auto-downloads on first `/asr`.
  Re-pull pinned commit: `huggingface.co/Systran/faster-whisper-medium` @
  `08e178d48790749d25932bbc082711ddcfdfbc4f`.

### 6. ettin cross-encoder — `cross-encoder/ettin-reranker-150m-v1`
- **Apache-2.0.** Base model `jhu-clsp/ettin-encoder-150m`. This is the **primary salience gate
  AND memory-rerank model** (one shared instance, the ⅓-VRAM win over the retired DeBERTa-large
  zeroshot gate). Loaded via `sentence_transformers.CrossEncoder` — requires `sentence-transformers`
  in the venv (pinned in requirements). Lives in the **`C:\` HF cache** (default `~/.cache`).
- Auto-downloads on first run. Re-pull pinned commit:
  `huggingface.co/cross-encoder/ettin-reranker-150m-v1` @ `025501c4e0f9bbeb4c5b198318e0089ff061cc14`.

### 7. emotion classifier — `SamLowe/roberta-base-go_emotions`
- **MIT.** Base model `roberta-base`; fine-tuned on the GoEmotions dataset (28 labels,
  multi-label sigmoid). The emotion layer's semantic affect signal (CPU, ~15 ms/pass). Lives in
  the **`C:\` HF cache**. Auto-downloads on first run. Re-pull pinned commit:
  `huggingface.co/SamLowe/roberta-base-go_emotions` @ `d75048347613a25d77de8cf6412eaae9fa7b26be`.

### 8. grounding NLI — `tasksource/deberta-small-long-nli`
- **Apache-2.0.** Base model `microsoft/deberta-v3-small`; 3-class NLI (entail/neutral/contradiction),
  1680-token window. Read-time memory grounding + write-time contradiction guard. Lives in the
  **`C:\` HF cache**. Auto-downloads on first run. Re-pull pinned commit:
  `huggingface.co/tasksource/deberta-small-long-nli` @ `9a77395d4d3751be9e2a69c4ae318491d9b3fffb`.

### 9a. llama.cpp native runtime
- MIT. Build **9305** (`63248fc3e`), CUDA 13.1, Clang 19.1.5, x86_64. Per MANIFEST §6a the
  `.exe` files are thin launchers — **ship the whole `llama.cpp` folder**, not just exes.
- Reproducible fetch: the two release zips are retained in the folder and hashed above —
  `llama-b9305-cuda13.1.zip` (151 MiB) + `cudart-cuda13.1.zip` (384 MiB). Upstream release:
  `https://github.com/ggml-org/llama.cpp/releases/tag/b9305`.
- ⚠️ **License split inside the folder:** llama.cpp's own binaries are MIT, but the bundled
  NVIDIA `cudart64_13.dll` / `cublas64_13.dll` / `cublasLt64_13.dll` are **#10 above — NVIDIA
  CUDA redistribution terms, not MIT.** "Ship the whole folder" is fine for personal/WiP copy,
  but a redistributor must honor NVIDIA's terms for those DLLs (or have the consumer fetch the
  CUDA runtime themselves). The MIT statement covers llama.cpp, not the CUDA DLLs it links.

---
_Verification method: SHA256 via `Get-FileHash` (and HF-cache blob names, which equal their
LFS SHA256 — self-verifying); GGUF source via embedded GGUF metadata; licenses + base models +
repo file lists confirmed against HuggingFace/GitHub model cards on 2026-06-04 and 2026-06-09._
