# Vivianna Architecture

_Work-in-progress architecture snapshot — 04 June 2026._

Vivianna is an experimental local AI companion prototype focused on low-latency interaction, persistent memory, emotional continuity, and modular orchestration on consumer hardware.

Active development began on **22 May 2026**. This document describes the project state as of **04 June 2026**, approximately **13 days into active development**.

No claims of novelty, exclusivity, or industry firsts are made. This architecture combines existing open-source components with custom orchestration, memory handling, routing, and interaction design.

---

## 1. High-Level Goal

The core goal is not to maximize benchmark scores.

The goal is to test how much conversational presence, continuity, and emotional coherence can be achieved with a fully local architecture running on consumer hardware.

Vivianna prioritizes:

- Fast response initiation
- Local-first execution
- Persistent memory
- Stable character continuity
- Speech interaction
- Modular doctrine / judgment layers
- Reversible experimentation through kill switches and configuration flags

The system is designed around perceived interaction quality rather than pure model scale.

---

## 2. Current Runtime Stack

Current validated stack:

| Layer | Current implementation |
|---|---|
| LLM runtime | llama.cpp server |
| Main model | Qwen3.5-9B GGUF |
| ASR | Faster-Whisper medium / CUDA / INT8 |
| TTS | Kokoro ONNX fp32 |
| Memory embeddings | FastEmbed / BGE small |
| Salience / judgment experiments | DeBERTa large zeroshot |
| Web retrieval | Optional Tavily / Exa / fetch pipeline |
| Platform | Windows 11 |
| Tested GPU | RTX 5070 12 GB |

The release assumes a local folder-based setup. Models and API keys are not bundled in the repository.

---

## 3. Architectural Overview

The runtime is organized as a layered local interaction loop:

```text
User input
  ↓
Keyboard or ASR
  ↓
Input loop
  ↓
Router / role / NLI classification
  ↓
Tool decision or local chat
  ↓
Memory retrieval + session summary + doctrine state
  ↓
Prompt assembly
  ↓
Qwen via llama.cpp
  ↓
Streaming output
  ↓
Post-stabilizer / cadence merge
  ↓
TTS queue
  ↓
Kokoro speech playback
  ↓
Memory commit / summary compression
```

The important design point is that Vivianna is not just a single prompt attached to a model. The model is surrounded by small orchestration layers that manage memory, routing, speech, state, confidence, and future judgment expansion.

---

## 4. Input Layer

Vivianna supports two input modes:

- Keyboard input
- Voice input through local ASR

The current ASR path uses Faster-Whisper through CTranslate2. The selected runtime configuration is medium / CUDA / INT8 because it offered the best accuracy-per-VRAM balance during testing.

Voice input is intentionally lazy-loaded and toggleable. This keeps startup lighter and allows keyboard fallback during debugging.

Relevant runtime concepts:

- `/voice` toggles microphone vs keyboard interaction
- `/asr` toggles local ASR engine loading
- ASR is local; cloud ASR is not part of the current required runtime
- Language detection is automatic for English / German code-switching experiments

---

## 5. Routing Layer

Before the main LLM response is generated, Vivianna passes input through routing logic.

The router is responsible for deciding whether the input should become:

- A normal local chat response
- A deterministic local tool call
- A time/date response
- A web retrieval task
- A direct `/read` fetch-and-speak operation
- A future sensitive-case / judgment path

This keeps simple deterministic tasks out of the main generative loop where possible.

The current routing design uses deterministic guards first, then NLI confidence bands for more ambiguous decisions.

---

## 6. Acknowledgement Layer

The acknowledgement layer is an input-side conversational timing mechanism.

Its purpose is to let Vivianna respond quickly with a small acknowledgement before a longer answer is ready. This supports the illusion of presence and reduces perceived latency.

The current implementation is Phase A:

- Deterministic
- One acknowledgement per turn
- Priority-based
- Reversible through configuration flags

Future versions may allow a small classifier or judgment model to choose acknowledgements more flexibly, but the current layer is intentionally simple and inspectable.

---

## 7. Memory Layer

Vivianna has both short-term and long-term memory mechanisms.

### Short-term continuity

Conversation history is kept in a rolling window. When the window grows too large, the system compresses prior exchanges into a session summary.

This prevents the prompt from growing without bound while preserving continuity across turns.

### Long-term semantic memory

Long-term memory uses semantic embeddings and retrieval.

The current system stores memory vectors and metadata locally, then retrieves the top relevant memories before response generation.

The design goal is not perfect autobiographical memory. The goal is to give the model enough relevant context to preserve continuity without repeatedly injecting a huge character prompt.

---

## 8. Lazy Character Injection

Earlier versions injected the full character prompt every turn.

The current design uses lazy character injection:

- The full character configuration is injected at session start.
- Later turns rely on session summary, memory context, recent history, and stabilizer cues.
- This reduces prompt size and improves llama.cpp cache behavior.

This is one of the core performance decisions in the current architecture.

The intended effect is to preserve identity continuity without paying the full prompt cost on every turn.

---

## 9. Stabilizer Layer

The stabilizer is a lightweight layer around generation that helps preserve identity, reduce obvious drift, and suppress unwanted reasoning artifacts.

It is not treated as a magic safety system.

It is a local corrective layer that helps keep output aligned with the current runtime state and interaction doctrine.

Current stabilizer goals:

- Reduce reasoning leakage
- Preserve character continuity
- Prevent obvious role collapse
- Keep responses within the intended interaction frame

---

## 10. Emotion Layer

The emotion layer tracks a single primary emotional state with decay.

It is currently used as a contextual signal rather than a full emotional simulator.

The purpose is to make Vivianna's response style more coherent across turns. For example, a serious or apologetic state should not instantly disappear if the next prompt is only mildly related.

Future versions may route emotion state into TTS voice/style selection.

---

## 11. Confidence Layer

The confidence layer provides a coarse estimate of how confidently Vivianna should answer.

It combines signals such as routing certainty and grounding strength.

The current goal is not mathematical truth calibration. The goal is behavioral shaping:

- high confidence → direct answer
- medium confidence → cautious answer
- low confidence → clarification or softer wording

This helps prevent the assistant from sounding equally certain in all situations.

---

## 12. Salience Layer

The salience layer is intended to decide which facts deserve memory storage or retrieval priority.

Current status:

- Configuration exists.
- Salience weighting is part of the memory design.
- DeBERTa large is planned / partially integrated as a semantic salience scorer.
- As of this snapshot, the salience judge is under investigation and should be treated as experimental.

Conceptually, this layer exists because not all facts are equally worth remembering.

Examples of high-salience memory:

- User preferences
- Project decisions
- Safety-relevant information
- Stable personal facts
- Repeated behavioral patterns

Examples of low-salience memory:

- One-off phrasing
- Temporary debugging noise
- Accidental test facts
- Non-actionable chatter

---

## 13. DeBERTa as a Doctrine / Judgment Component

DeBERTa is not included merely as an emotion classifier.

The architectural intent is broader: use a strong encoder-style model as a reusable judgment layer for multiple small decisions.

Possible responsibilities include:

- Salience scoring
- Memory commit decisions
- Retrieval reranking
- Role / boundary classification
- Tool usefulness reassessment
- Sensitive-case triage
- Confidence support

The design reason is amortization. If one model is already resident or available, several small doctrine heads or classification prompts may be routed through it rather than spawning separate models for every signal.

This is still experimental and should not be presented as complete.

---

## 14. Qwen Guardrail / Future Guardrail Placeholder

A future version may use Qwen Guardrail or a similar dedicated guardrail model as a placeholder or replacement for some doctrine and judgment functions once it is mature enough for the project.

Potential future uses:

- Sensitive-case classification
- Safety and boundary judgment
- Memory commit review
- Tool-use gating
- Refusal / caution decisions

This is not claimed as implemented in the current release.

It is an architectural recommendation and future expansion point.

---

## 15. Web Retrieval Layer

Web retrieval is optional and modular.

The current design separates:

- explicit user-triggered reading
- search / retrieval provider selection
- text extraction
- cleanup
- speech-safe output formatting

API keys are not bundled. Web providers can be omitted without preventing the local core from running.

The long-term goal is not to make the model blindly trust web results. The goal is to provide controlled retrieval paths that can later be judged, filtered, summarized, and cited.

---

## 16. Output Layer

The output path is streaming-first.

The model streams text from llama.cpp. Output then passes through:

- stream stabilization
- reasoning-bleed suppression
- cadence / chunk merging
- TTS priority queue
- Kokoro synthesis
- audio playback

This keeps text output responsive while allowing spoken output to follow in manageable chunks.

The current TTS path uses Kokoro ONNX fp32. It is CPU-based in the current runtime to preserve GPU headroom.

---

## 17. Memory Commit and Compression

After output, the system may update memory and session state.

Two separate concepts matter:

### Memory commit

The system decides whether anything from the exchange should be stored as long-term memory.

### Session compression

The system periodically compresses conversation history into a running summary to prevent the rolling window from growing too large.

These are related but not identical.

A conversation can be compressed without committing new long-term memory.

---

## 18. Runtime Toggles and Kill Switches

Most major layers are controlled through configuration flags.

This is intentional.

The project is still experimental, so individual subsystems should be easy to disable for debugging, A/B testing, or rollback.

Examples of configurable layers:

- TTS
- ASR
- emotion layer
- salience layer
- confidence layer
- role layer
- acknowledgement layer
- lazy character injection

This is part of the architecture, not an accident. The system is designed to be inspectable and reversible.

---

## 19. Current Known Architectural Limitations

This release is not a polished general-purpose assistant.

Known limitations include:

- Tested on one hardware configuration only
- Several paths are still Windows- and F:\AI-specific
- Salience judge is currently under investigation
- Some doctrine layers are experimental
- Web retrieval quality is still being improved
- German TTS is deferred / experimental
- Memory confidentiality controls are not complete
- Not intended for unattended safety-critical use

---

## 20. Diagram

The current doctrine / orchestration diagram is provided as a visual aid.

If included in the repository, place it for example at:

```text
docs/vivianna_architecture_diagram.png
```

and reference it from this document:

```markdown
![Vivianna architecture diagram](docs/vivianna_architecture_diagram.png)
```

The diagram should be treated as a conceptual map, not a guarantee that every future / placeholder box is implemented today.

---

## 21. Verification and Provenance

For runtime inventory and artifact provenance, see:

- `MANIFEST.md`
- `PROVENANCE.md`
- `RUNTIME_STATUS_2026-06-04.md`

This architecture document explains the design. It does not replace installation instructions or provenance verification.

Some components are implemented. Some are experimental. Some are future placeholders. The current release should preserve that distinction clearly.

---

## 22. Summary

Vivianna is best understood as a local orchestration experiment around a small resident language model.

The central architectural bet is that perceived companion quality can be improved through:

- low-latency local inference
- persistent memory
- summary compression
- stable identity cues
- speech interaction
- modular doctrine layers
- careful routing and output handling

rather than relying only on a larger model or a larger prompt.

This is a work-in-progress snapshot, not a finished product.
