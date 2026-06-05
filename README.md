[![Vivianna Demo](https://img.youtube.com/vi/ZfMixvZ_T3g/0.jpg)](https://www.youtube.com/watch?v=ZfMixvZ_T3g)

# Vivianna (Work In Progress)

I fell into this rabbit hole as someone with no CS Background and no plan how to do anything. This entire project is the work of 3 frontier LLMs and a simple human trying to create something useful for the household. This is day 14 on an RTX 5070 and it runs end2end on my system. Story of how this came to be will follow within next days including a simple clip to show the terminal. Visualization is not included since the orchestration needs to fully work first. Adding additional failure modes would fall out of my capability to pilot Claude Code. So maximum credits to Qwen as base model, frontier Model as Audit, GPT for his strict RLHF verdicts and Claude for doing the hardwork.

## What is Vivianna?

Vivianna is an experimental local AI companion prototype focused on:

- Low-latency local interaction
- Persistent memory
- Emotional continuity
- Speech input and output
- Modular cognitive doctrine layers
- Fully local operation without mandatory cloud services

Active development began on **22 May 2026**.

This repository represents the state of the project as of **04 June 2026**, approximately **13 days into active development**.

The project is provided as a work-in-progress snapshot for educational, research, and hobbyist purposes.

No claims of novelty, exclusivity, or industry firsts are made. The project combines existing open-source components with custom orchestration, memory, routing, and interaction design choices.

---

## Current Status

Validated on:

- Windows 11
- AMD Ryzen 7 9700X
- NVIDIA RTX 5070 12 GB
- 32 GB DDR5 RAM

Current runtime stack:

- Qwen3.5-9B GGUF via llama.cpp
- Faster-Whisper (medium / CUDA / INT8)
- Kokoro ONNX TTS
- FastEmbed memory retrieval
- DeBERTa-based salience experiments
- Local memory persistence
- Optional web retrieval providers

---

## Tested Hardware

The current release has been validated on the following system:

- AMD Ryzen 7 9700X
- NVIDIA RTX 5070 12 GB
- 32 GB DDR5 RAM
- Windows 11

This project has currently been tested on a single hardware configuration only. Other systems may require path adjustments, CUDA configuration changes, model relocation, or additional troubleshooting.

---

## What Ships In This Repository

Included:

- Source code
- Configuration files
- Launch scripts
- Runtime documentation
- Architecture documentation
- Requirements list

Not included:

- LLM weights
- ASR model weights
- TTS model weights
- Hugging Face cache artifacts
- API keys

These must be downloaded separately from their original sources.

---

## Quick Start
[for more detailed install instructions look at this](INSTALL.md)
### 1. Create Python Environment

Install Python and create a virtual environment.

```bash
pip install -r requirements_full_2026-06-04.txt
```

### 2. Install llama.cpp

Download the version documented in `PROVENANCE.md`.
Extract to:

```text
F:\AI\llama.cpp
```

or adjust paths accordingly.

### 3. Download Required Models

See:

- PROVENANCE.md
- MANIFEST.md

Required runtime models:

- Qwen3.5-9B-UD-Q4_K_XL.gguf
- Qwen3.5-9B-mmproj-BF16.gguf
- kokoro-v1.0.onnx
- voices-v1.0.bin
- BGE Small Embedder
- Faster-Whisper Medium
- DeBERTa Large Zeroshot v2.0

Place them in the documented locations.

### 4. Optional API Keys

The following integrations are optional:

- Tavily
- Exa

Place API keys in:

```text
F:\AI\API keys\
```

If omitted, corresponding features will be unavailable.

### 5. Start Vivianna

Run:

```text
run_vivianna.bat
```

The launcher starts llama.cpp, waits for readiness, and launches Vivianna.

---

## Known Limitations

This is a work-in-progress prototype.

Current known issues include:

- Salience judge currently dormant. (for immediate fix I recommend looking at https://qwen.ai/blog?id=qwen3guard)
- Hardcoded Windows paths remain in several components
- DeBERTa cache currently lives inside Hugging Face cache locations
- German TTS path remains experimental
- Retrieval quality is still being actively improved

---

## Design Philosophy

[Diagram to show how it is designed to work in future](mermaid-diagram.png)

[Design Philosophy in Detail](Vivianna%20Doctrine.md)

The goal is not to maximize benchmark scores.

The goal is to explore how much conversational continuity, memory persistence, emotional coherence, and identity stability can be achieved using a fully local architecture running on consumer hardware.

The project prioritizes:

- Local execution
- Fast response initiation
- Persistent identity
- Memory continuity
- Modular experimentation

over benchmark chasing or frontier-scale model sizes.

---

## Repository Philosophy

This release prioritizes transparency over polish.

Documentation includes known limitations, unresolved bugs, verification boundaries, and active work items whenever possible.

---

## Verification

See:

- MANIFEST.md
- PROVENANCE.md

for dependency inventory, artifact provenance, licenses, and verification status.

Some artifacts are fully verified. Others are documented but not independently verified. Verification boundaries are explicitly stated in the provenance documentation.

---

## License

This repository — the original source code, configuration, and documentation — is licensed
under the **MIT License**. See `LICENSE`.

It **depends on** separately-downloaded third-party models and runtimes (Qwen, Kokoro,
faster-whisper, BGE, DeBERTa, llama.cpp, and the NVIDIA CUDA runtime) that are **not included
in this repository** and retain their own licenses. Those licenses — including the NVIDIA CUDA
redistribution terms — are documented per artifact in `PROVENANCE.md`. The MIT license here
covers only the code in this repo, not the components you download separately.
