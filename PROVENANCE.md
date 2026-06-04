# Artifact Provenance — Vivianna WiP free-to-copy release

_Generated 2026-06-04. Companion to MANIFEST §6 (file inventory). This file answers the
three questions a copier actually needs: **where each artifact came from, that they got the
right bytes (SHA256), and whether redistribution is allowed.**_

## Headline finding
**All 7 model weights + the llama.cpp runtime are Apache-2.0 or MIT — legally redistributable
as blobs.** No weight is license-restricted, so nothing is *forced* to be fetch-only;
`fetch-only` below is a size/convenience recommendation only, never a legal requirement.
**The one exception is the NVIDIA CUDA runtime (#9):** its `cudart`/`cublas` DLLs are *not*
open-source — they are redistributable, but only under **NVIDIA's CUDA redistribution terms**
(the CUDA Toolkit EULA's redistributable-components clause). Legally shippable with attribution
to those terms; simplest clean path is **fetch-only** (let the consumer pull NVIDIA's own
`cudart` redistributable, or take it via the llama.cpp CUDA release). The cloned-voice path is
gone (`victoria.npy` deleted 2026-06-04), so there is no voice-rights exposure either.

## How to read this
- **SHA256** is over the exact file at the listed path. For the three HuggingFace-cache
  artifacts (bge, DeBERTa, faster-whisper) the hash is of the **primary weight file** and —
  because HF stores LFS objects under their own SHA256 — that hash equals the on-disk blob
  name, i.e. it is self-verifying against HF. The exact-revision anchor for those is the
  **pinned commit hash** in the source URL (re-pulling that commit yields bit-identical files).
- All hashes are lowercase SHA256.

## Provenance table

| # | Artifact (local name) | Source (repo @ revision) | License | SHA256 (primary file) | Ship-as |
|---|---|---|---|---|---|
| 1 | `Qwen3.5-9B-UD-Q4_K_XL.gguf` (5.56 GiB) | `huggingface.co/unsloth/Qwen3.5-9B-GGUF` @ `main` (file `Qwen3.5-9B-UD-Q4_K_XL.gguf`) | Apache-2.0 | `6f5d30666c2d8ae16a306e616d95341dcf3cc46810df84d7e6f5a7d1e4c1b293` | blob (or fetch) |
| 2 | `Qwen3.5-9B-mmproj-BF16.gguf` (879 MiB) | `huggingface.co/unsloth/Qwen3.5-9B-GGUF` @ `main` (**upstream file `mmproj-BF16.gguf`**) | Apache-2.0 | `d89c4bc142d02ed64aeed5c0a358bdead9109f21f4ada03a6b2df17a1aa94d9e` | blob (or fetch) |
| 3 | `kokoro-v1.0.onnx` fp32 (310 MiB) | `github.com/thewh1teagle/kokoro-onnx` release `model-files-v1.0` | Apache-2.0 (weights) | `7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5` | blob |
| 4 | `voices-v1.0.bin` (26.9 MiB) | `github.com/thewh1teagle/kokoro-onnx` release `model-files-v1.0` | Apache-2.0 (weights) | `bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d` | blob |
| 5 | `models--qdrant--bge-small-en-v1.5-onnx-q` (folder ~65 MB; weight `model_optimized.onnx` 63.4 MiB) | `huggingface.co/qdrant/bge-small-en-v1.5-onnx-q` @ `52398278842ec682c6f32300af41344b1c0b0bb2` | Apache-2.0 | `51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431` | blob (or fetch) |
| 6 | `models--MoritzLaurer--deberta-v3-large-zeroshot-v2.0` (folder 841 MB; weight `model.safetensors` 830 MiB) | `huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0` @ `cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2` | MIT | `2031ec34340911b2cecf4f95f5e24db91b2a5d7ea0fa1e704e9cd6d61585e477` | fetch (on `C:\`) |
| 7 | `models--Systran--faster-whisper-medium` (folder 1.46 GB; weight `model.bin` 1.42 GiB) | `huggingface.co/Systran/faster-whisper-medium` @ `08e178d48790749d25932bbc082711ddcfdfbc4f` | MIT | `9b45e1009dcc4ab601eff815b61d80e60ce3fd8c74c1a14f4a282258286b51ae` | blob (or fetch) |
| 8 | `llama.cpp` runtime — build **9305** (`63248fc3e`), CUDA 13.1 | `github.com/ggml-org/llama.cpp` release `b9305`; retained asset `llama-b9305-cuda13.1.zip` | MIT | `882df5e511259e4976c137b06ff56a2c0aa6433811b350a26e7e828aea5beea6` (zip) | fetch (whole folder, see §8a) |
| 9 | CUDA 13.1 runtime DLLs | NVIDIA CUDA runtime, bundled with the llama.cpp CUDA release; retained asset `cudart-cuda13.1.zip` | **NVIDIA CUDA redistribution terms** (NOT open-source — see note) | `f96935e7e385e3b2d0189239077c10fe8fd7e95690fea4afec455b1b6c7e3f18` (zip) | fetch (with #8) |

## Per-artifact fetch instructions / notes

### 1–2. Qwen3.5-9B GGUFs — `unsloth/Qwen3.5-9B-GGUF`
- Source confirmed from the GGUF's own embedded metadata, not guessed:
  `general.quantized_by = Unsloth`, `quantize.imatrix.file = Qwen3.5-9B-GGUF/imatrix_unsloth.gguf`,
  `general.license = apache-2.0`, base model `Qwen/Qwen3.5-9B`. Repo license on HF = `apache-2.0`.
- **Rename gotcha:** upstream the projector is `mmproj-BF16.gguf` (922 MB on HF). Our local
  copy was renamed to `Qwen3.5-9B-mmproj-BF16.gguf`. A copier pulling from HF gets the
  unprefixed name and must rename (or adjust the launch flag).
- Fetch:
  - `https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-UD-Q4_K_XL.gguf`
  - `https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/mmproj-BF16.gguf`
- Revision: these two live under `F:\AI\Models` as plain files (not in HF-cache layout), so
  there is no cached commit pin. The **SHA256 above is the byte anchor** — verify after pull.
  Upstream `main` may advance; if hashes differ, the repo was re-uploaded.

### 3–4. Kokoro TTS — `thewh1teagle/kokoro-onnx`
- Model weights are Apache-2.0 (derived from `hexgrad/Kokoro-82M`); the kokoro-onnx project
  code is MIT. For these two **weight** blobs the governing license is Apache-2.0.
- Fetch (release tag pins the revision):
  - `https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx`
  - `https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin`
- Runtime uses stock voices `af_heart` (EN) + `zf_xiaobei` (ZH) from `voices-v1.0.bin`.

### 5. bge-small embedder — `qdrant/bge-small-en-v1.5-onnx-q`
- Apache-2.0 ONNX/quantized port of `BAAI/bge-small-en-v1.5`. Lives in the Vivianna data dir
  (`F:\AI\Vivianna\data\fastembed_cache`), **not** `F:\AI\Models`.
- Re-pull pinned commit: `huggingface.co/qdrant/bge-small-en-v1.5-onnx-q` @
  `52398278842ec682c6f32300af41344b1c0b0bb2`.

### 6. DeBERTa salience gate — `MoritzLaurer/deberta-v3-large-zeroshot-v2.0`
- MIT (foundation model). **Lives on `C:\` (`%USERPROFILE%\.cache\huggingface\hub`), not `F:\`.**
  An F:\-scoped package must either copy this folder explicitly or let the consumer re-pull.
  The relocate/bundle decision is still an open MANIFEST item.
- Re-pull pinned commit: `huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0` @
  `cf44676c28ba7312e5c5f8f8d2c22b3e0c9cdae2`.

### 7. faster-whisper ASR — `Systran/faster-whisper-medium`
- MIT. CTranslate2 model; runtime config = medium / cuda / int8. Re-pull pinned commit:
  `huggingface.co/Systran/faster-whisper-medium` @ `08e178d48790749d25932bbc082711ddcfdfbc4f`.

### 8a. llama.cpp native runtime
- MIT. Build **9305** (`63248fc3e`), CUDA 13.1, Clang 19.1.5, x86_64. Per MANIFEST §6a the
  `.exe` files are thin launchers — **ship the whole `F:\AI\llama.cpp` folder**, not just exes.
- Reproducible fetch: the two release zips are retained in the folder and hashed above —
  `llama-b9305-cuda13.1.zip` (151 MiB) + `cudart-cuda13.1.zip` (384 MiB). Upstream release:
  `https://github.com/ggml-org/llama.cpp/releases/tag/b9305`.
- ⚠️ **License split inside the folder:** llama.cpp's own binaries are MIT, but the bundled
  NVIDIA `cudart64_13.dll` / `cublas64_13.dll` / `cublasLt64_13.dll` are **#9 above — NVIDIA
  CUDA redistribution terms, not MIT.** "Ship the whole folder" is fine for personal/WiP copy,
  but a redistributor must honor NVIDIA's terms for those DLLs (or have the consumer fetch the
  CUDA runtime themselves). The MIT statement covers llama.cpp, not the CUDA DLLs it links.

---
_Verification method: SHA256 via `Get-FileHash`; GGUF sources via embedded GGUF metadata;
licenses + repo file lists confirmed against HuggingFace/GitHub on 2026-06-04. HF-cache
weight hashes equal their LFS blob names (self-verifying)._
