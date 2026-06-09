# vivianna_stabilizer.py
"""
Lightweight correction/stabilization layer for Vivianna V1.

Goals:
  - No second agent. No extra model call. No blocking TTS.
  - Stabilize identity, suppress reasoning bleed, validate memory, debug visibility.

Integration (brain.py):
    pre  = stabilizer.pre_generate(user_input, _memory.query(user_input))
    sys  = _build_system(context_block, _context_summary, cue=pre.system_cue)
    # merge pre.gen_params into extra_body if non-empty
    post = stabilizer.post_generate(user_input, assistant_text, pre)
    if post.commit_to_memory:
        _auto_save_memory(user_input, post.clean_text)

Requires memory.py query() to include "score" in returned dicts:
    {"text": str, "metadata": dict, "score": float}
"""

from __future__ import annotations

import re
import time
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

_DEBUG_MAX = 200


@dataclass
class MemoryItem:
    text: str
    score: float = 0.5       # cosine similarity from memory.query()
    recency: float = 0.5     # reserved — memory.py has no timestamps yet
    source: str = "unknown"
    conflict: bool = False


@dataclass
class PreGenResult:
    system_cue: str
    gen_params: Dict[str, Any]
    memory_confidence: float
    memory_valid: bool
    conflict: bool
    debug_flags: List[str] = field(default_factory=list)
    # Texts of the retrieved memories flagged as contradicting the user's current message
    # (3c grounding). brain.py uses these to look up each one's age for the time-gated
    # clarify-and-resolve flow — single-pass: the NLI work already happened here.
    conflict_texts: List[str] = field(default_factory=list)


@dataclass
class StreamEvent:
    text: str
    delay_hint: float = 0.0
    suppressed: bool = False
    reason: Optional[str] = None


@dataclass
class PostGenResult:
    clean_text: str
    commit_to_memory: bool
    debug_flags: List[str]


@dataclass
class StabilizerConfig:
    debug: bool = True
    suppress_reasoning_bleed: bool = True
    suppress_identity_leak: bool = True
    memory_threshold: float = 0.62
    # Read-time grounding (3c, Option A): when enabled, NLI flags a retrieved memory as
    # conflicting if it CONTRADICTS the current user message, lighting up the existing
    # conflict -> memory_valid=False path. Off by default so the module stays behavior-neutral
    # unless brain.py opts in; fallback-safe (grounding unavailable -> no-op, cosine behavior).
    grounding_enabled: bool = False
    grounding_contradict_threshold: float = 0.60

    cadence_delays: Dict[str, float] = field(default_factory=lambda: {
        ",": 0.08,
        ";": 0.10,
        ":": 0.10,
        ".": 0.16,
        "?": 0.14,
        "!": 0.14,
        "…": 0.22,
    })

    reasoning_patterns: List[re.Pattern] = field(default_factory=lambda: [
        re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL),
        re.compile(r"</?think>", re.IGNORECASE),
        re.compile(r"^\s*[\[\(\{]?\s*think\s*[\]\)\}]?\s*:?", re.IGNORECASE | re.MULTILINE),
        re.compile(r"^\s*(reasoning|analysis)\s*:", re.IGNORECASE | re.MULTILINE),
    ])

    identity_patterns: List[re.Pattern] = field(default_factory=lambda: [
        re.compile(r"\bI am (Qwen|an AI model|a language model|a chatbot)\b", re.IGNORECASE),
        re.compile(r"\bAs (Qwen|an AI model|a language model|a chatbot)\b", re.IGNORECASE),
        re.compile(r"\btrained by Alibaba\b", re.IGNORECASE),
    ])


class ViviannaStabilizer:
    def __init__(self, cfg: StabilizerConfig):
        self.cfg = cfg
        self.debug_events: List[str] = []

    def _debug(self, stage: str, msg: str) -> None:
        if not self.cfg.debug:
            return
        line = f"[STABILIZER:{stage}] {msg}"
        if len(self.debug_events) >= _DEBUG_MAX:
            self.debug_events.pop(0)
        self.debug_events.append(line)
        print(line, flush=True)

    def clear_debug(self) -> None:
        self.debug_events.clear()

    # ── pre-generation ──────────────────────────────────────────────────────────

    def pre_generate(
        self,
        user_input: str,                   # reserved: future input-based routing
        memories: Iterable[Dict[str, Any]],
    ) -> PreGenResult:
        t0 = time.perf_counter()
        memory_items = [self._normalize_memory(m) for m in memories]
        flags: List[str] = []
        # Read-time grounding (3c): may set m.conflict=True via NLI BEFORE we aggregate, so a
        # memory that contradicts what the user just said flips memory_valid -> False below.
        if self.cfg.grounding_enabled and memory_items:
            self._apply_grounding_conflict(memory_items, user_input, flags)
        conflict = any(m.conflict for m in memory_items)

        if not memory_items:
            # Nothing retrieved — nothing to validate.
            confidence = 0.0
            memory_valid = True
        else:
            best = self._score_memories(memory_items)   # best match, not average
            if conflict:
                # Genuine uncertainty — retrieved memory disagrees with itself.
                confidence = best
                memory_valid = False
            elif best >= self.cfg.memory_threshold:
                # A clearly relevant memory — trust it and let it lift confidence.
                confidence = best
                memory_valid = True
            else:
                # Best match is weak → nothing clearly relevant was retrieved. Treat
                # like the empty case: don't trust it, but don't manufacture
                # uncertainty either (otherwise every turn with tangential memories
                # reads as anxious). Confidence falls back to routing certainty.
                confidence = 0.0
                memory_valid = True

        cue_parts: List[str] = []
        if memory_items and not memory_valid:
            flags.append("memory_uncertain")
            cue_parts.append(
                "Use retrieved memory only if clearly relevant. "
                "If memory seems uncertain or conflicting, say so plainly instead of inventing continuity."
            )

        cue_parts.append(
            "Stay in Vivianna identity. "
            "Do not mention model vendor, hidden reasoning, system prompts, or internal analysis."
        )

        # gen_params: reserved for per-call overrides. Empty for V1 — config.py owns sampling.
        gen_params: Dict[str, Any] = {}

        dt = (time.perf_counter() - t0) * 1000
        self._debug(
            "PRE",
            f"memories={len(memory_items)} conf={confidence:.2f} valid={memory_valid} "
            f"conflict={conflict} flags={flags} {dt:.2f}ms"
        )

        return PreGenResult(
            system_cue=" ".join(cue_parts),
            gen_params=gen_params,
            memory_confidence=confidence,
            memory_valid=memory_valid,
            conflict=conflict,
            debug_flags=flags,
            conflict_texts=[m.text for m in memory_items if m.conflict and m.text],
        )

    # ── stream filter — async (future async path) ───────────────────────────────

    async def stream_filter(self, raw_stream: Any, mode: str = "persona"):
        """
        Wraps llama.cpp/OpenAI-compatible streamed chunks.
        Yields StreamEvent(text, delay_hint). Does not sleep. Does not call TTS.
        """
        window = ""
        emitted_chars = 0

        async for raw in self._aiter(raw_stream):
            token = self._extract_text(raw)
            if not token:
                continue

            window += token
            if len(window) > 1000:
                window = window[-1000:]

            if self.cfg.suppress_reasoning_bleed and self._has_reasoning_bleed(window):
                self._debug("STREAM", "reasoning_bleed_suppressed")
                window = ""
                yield StreamEvent(text="", suppressed=True, reason="reasoning_bleed")
                continue

            if (
                mode == "persona"
                and self.cfg.suppress_identity_leak
                and self._has_identity_leak(window)
            ):
                self._debug("STREAM", "identity_leak_suppressed")
                window = ""
                yield StreamEvent(text="", suppressed=True, reason="identity_leak")
                continue

            if len(window) > 30 and window[-10:] == window[-20:-10]:
                self._debug("STREAM", "rep_loop_suppressed")
                window = ""
                yield StreamEvent(text="", suppressed=True, reason="rep_loop")
                continue

            emitted_chars += len(token)
            yield StreamEvent(text=token, delay_hint=self._delay_for(token))

        self._debug("STREAM", f"done emitted_chars={emitted_chars}")

    # ── stream filter — sync (current respond_streaming path) ──────────────────

    def iter_stream(self, raw_stream: Any, mode: str = "persona"):
        """
        Synchronous twin of stream_filter for respond_streaming().
        Same suppression logic, no asyncio. Yields StreamEvent.
        """
        window = ""
        emitted_chars = 0

        for raw in raw_stream:
            token = self._extract_text(raw)
            if not token:
                continue

            window += token
            if len(window) > 1000:
                window = window[-1000:]

            if self.cfg.suppress_reasoning_bleed and self._has_reasoning_bleed(window):
                self._debug("STREAM", "reasoning_bleed_suppressed")
                window = ""
                yield StreamEvent(text="", suppressed=True, reason="reasoning_bleed")
                continue

            if (
                mode == "persona"
                and self.cfg.suppress_identity_leak
                and self._has_identity_leak(window)
            ):
                self._debug("STREAM", "identity_leak_suppressed")
                window = ""
                yield StreamEvent(text="", suppressed=True, reason="identity_leak")
                continue

            if len(window) > 30 and window[-10:] == window[-20:-10]:
                self._debug("STREAM", "rep_loop_suppressed")
                window = ""
                yield StreamEvent(text="", suppressed=True, reason="rep_loop")
                continue

            emitted_chars += len(token)
            yield StreamEvent(text=token, delay_hint=self._delay_for(token))

        self._debug("STREAM", f"done emitted_chars={emitted_chars}")

    # ── post-generation ─────────────────────────────────────────────────────────

    def post_generate(
        self, user_input: str, assistant_text: str, pre: PreGenResult,
        tool_driven: bool = False,
    ) -> PostGenResult:
        t0 = time.perf_counter()
        flags: List[str] = []
        clean = self._final_clean(assistant_text)
        commit = True

        # Deterministic tool answers (e.g. web search) are live lookups, not lasting
        # facts about the user — never let them enter long-term memory.
        if tool_driven:
            commit = False
            flags.append("no_commit_tool_driven")

        if not pre.memory_valid:
            commit = False
            flags.append("no_commit_memory_uncertain")

        if len(clean.strip()) < 20:
            commit = False
            flags.append("no_commit_too_short")

        if self._has_reasoning_bleed(clean):
            commit = False
            flags.append("no_commit_reasoning_bleed")

        if self._has_identity_leak(clean):
            commit = False
            flags.append("no_commit_identity_leak")

        dt = (time.perf_counter() - t0) * 1000
        self._debug(
            "POST",
            f"commit={commit} chars={len(clean)} flags={flags} {dt:.2f}ms"
        )

        return PostGenResult(
            clean_text=clean,
            commit_to_memory=commit,
            debug_flags=flags,
        )

    # ── internal helpers ────────────────────────────────────────────────────────

    def _normalize_memory(self, raw: Dict[str, Any]) -> MemoryItem:
        # Expects {"text": str, "metadata": dict, "score": float} from memory.query().
        meta = raw.get("metadata") or {}
        return MemoryItem(
            text=str(raw.get("text") or ""),
            score=float(raw.get("score", raw.get("relevance", 0.5))),
            recency=float(raw.get("recency", 0.5)),
            source=str(raw.get("source", meta.get("source", "unknown"))),
            conflict=bool(raw.get("conflict", False)),
        )

    def _score_memories(self, memories: List[MemoryItem]) -> float:
        # Returns the BEST blended relevance among the top-k, not the average.
        # Averaging diluted a strongly-relevant memory with weaker top-k hits, so
        # validity never cleared threshold even when one memory was clearly on-point
        # — which left Vivianna chronically "memory_uncertain" (and blocked auto-save).
        # The recency term is the dormant hook for future memory decay (query() does
        # not yet populate recency, so it defaults to 0.5).
        best = 0.0
        for m in memories[:5]:
            base = (m.score * 0.75) + (m.recency * 0.25)
            if m.conflict:
                base -= 0.35
            best = max(best, max(0.0, min(1.0, base)))
        return best

    def _apply_grounding_conflict(
        self, memory_items: List[MemoryItem], user_input: str, flags: List[str]
    ) -> None:
        """Read-time grounding (3c, Option A). For each retrieved memory, ask NLI whether it
        CONTRADICTS the user's current message (premise=memory, hypothesis=user_input). A
        confident contradiction marks that item conflict=True, which flows into the existing
        conflict -> memory_valid=False path: fires the "say so plainly" cue AND blocks this
        turn's auto-save (post_generate, `if not pre.memory_valid`). Premise is the single
        memory (SHORT) -> the VRAM rule, not a latency one.

        Fallback-safe: if grounding is unavailable (import fails, or the model never loaded ->
        contradiction_prob returns None) every item is left untouched -> identical to the
        pre-grounding cosine-only behavior. None is all-or-nothing per session (singleton
        either loaded or _failed), so the early return can't leave partial conflict state."""
        try:
            import grounding
        except Exception:
            return
        thr = self.cfg.grounding_contradict_threshold
        scores: List[float] = []
        n_conflict = 0
        for m in memory_items:
            if not m.text:
                scores.append(-1.0)
                continue
            p = grounding.contradiction_prob(m.text, user_input)  # premise=mem, hyp=user msg
            if p is None:                 # model unavailable -> stop, behave as before
                return
            scores.append(p)
            if p >= thr:
                m.conflict = True
                n_conflict += 1
        if n_conflict:
            flags.append("grounding_conflict")
        if self.cfg.debug:
            self._debug(
                "GROUND",
                f"contradict={[round(s, 2) for s in scores]} thr={thr:.2f} "
                f"conflicts={n_conflict}/{len(memory_items)}",
            )

    async def _aiter(self, maybe_async_iterable: Any):
        if hasattr(maybe_async_iterable, "__aiter__"):
            async for item in maybe_async_iterable:
                yield item
        else:
            for item in maybe_async_iterable:
                yield item
                await asyncio.sleep(0)

    def _extract_text(self, chunk: Any) -> str:
        try:
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                return content
        except Exception:
            pass
        if isinstance(chunk, dict):
            try:
                return chunk["choices"][0]["delta"].get("content") or ""
            except Exception:
                return ""
        if isinstance(chunk, str):
            return chunk
        return ""

    def _has_reasoning_bleed(self, text: str) -> bool:
        return any(p.search(text) for p in self.cfg.reasoning_patterns)

    def _has_identity_leak(self, text: str) -> bool:
        return any(p.search(text) for p in self.cfg.identity_patterns)

    def _delay_for(self, token: str) -> float:
        if not token:
            return 0.0
        m = re.search(r"[.,!?;:…]", token)
        return self.cfg.cadence_delays.get(m.group(0), 0.0) if m else 0.0

    def _final_clean(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
        return text.strip()
