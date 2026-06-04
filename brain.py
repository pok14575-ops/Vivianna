import re
import os
import json
import time
import threading
from openai import OpenAI
from tts_preprocess import clean_for_tts
from output_bus import emit as tts_emit
import runtime_state
from config import (
    MODEL_NAME, SYSTEM_PROMPT, MAX_EXCHANGES, TEMPERATURE, MAX_TOKENS,
    SHOW_REASONING, BASE_URL, API_KEY, TOP_K, TOP_P, MIN_P,
    REPEAT_PENALTY, FREQUENCY_PENALTY, PRESENCE_PENALTY, DRY_MULTIPLIER,
    PROACTIVE_MEMORY, SUMMARY_TRIGGER, HISTORY_PATH,
    EMOTION_LAYER_ENABLED, EMOTION_DEBUG,
    SALIENCE_LAYER_ENABLED, SALIENCE_DEBUG, SALIENCE_STORE_THRESHOLD,
    SALIENCE_MODEL_ENABLED, SALIENCE_MODEL_NAME, SALIENCE_MODEL_DEVICE,
    CONFIDENCE_LAYER_ENABLED, CONFIDENCE_DEBUG,
    ROLE_LAYER_ENABLED, ROLE_DEBUG,
    ACK_LAYER_ENABLED, ACK_DEBUG, ACK_MODE, ACK_NEUTRAL_FALLBACK_THRESHOLD,
    TIME_AWARENESS_ENABLED,
    CHARACTER_SESSION_START_ONLY,
)
from tool_registry import TOOLS
from tools_lang import t
from tools_time import current_time_context
from memory import MemoryManager
from vivianna_stabilizer import ViviannaStabilizer, StabilizerConfig
from emotion_layer import EmotionLayer, EmotionConfig
from salience_layer import SalienceLayer, SalienceConfig
from confidence_layer import ConfidenceLayer, ConfidenceConfig
from role_layer import RoleLayer, RoleConfig
from ack_coordinator import AckCoordinator, AckConfig, DeterministicGate

# ── TEMP latency instrumentation (remove this block after profiling) ──────────
# Flips per-stage perf_counter timings on. Set False (or delete the block + every
# `with _stage(...)` use) to disable. Prints `[TIMING] <stage>: <secs>` per turn.
_STAGE_TIMING = True

class _stage:
    """Context manager that prints a stage's wall-clock duration when _STAGE_TIMING."""
    __slots__ = ("name", "_t")
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self._t = time.perf_counter()
        return self
    def __exit__(self, *exc):
        if _STAGE_TIMING:
            print(f"[TIMING] {self.name}: {time.perf_counter() - self._t:.3f}s", flush=True)
        return False
# ──────────────────────────────────────────────────────────────────────────────

client     = OpenAI(base_url=BASE_URL, api_key=API_KEY)
stabilizer = ViviannaStabilizer(StabilizerConfig())
emotion    = EmotionLayer(EmotionConfig(debug=EMOTION_DEBUG)) if EMOTION_LAYER_ENABLED else None
salience   = SalienceLayer(SalienceConfig(
    debug=SALIENCE_DEBUG,
    use_model=SALIENCE_MODEL_ENABLED,
    model_name=SALIENCE_MODEL_NAME,
    model_device=SALIENCE_MODEL_DEVICE,
)) if SALIENCE_LAYER_ENABLED else None
confidence = ConfidenceLayer(ConfidenceConfig(debug=CONFIDENCE_DEBUG)) if CONFIDENCE_LAYER_ENABLED else None
role       = RoleLayer(RoleConfig(debug=ROLE_DEBUG)) if ROLE_LAYER_ENABLED else None

# Ack coordinator (Tiered Cognition Doctrine, Phase A). Input-side, deterministic.
# emit_fn=tts_emit routes the ack through output_bus' "acknowledgement" priority slot
# (speak_priority), so it speaks ahead of the answer and buys time for generation.
if ACK_LAYER_ENABLED:
    ack_coord = AckCoordinator(
        cfg=AckConfig(mode=ACK_MODE, debug=ACK_DEBUG,
                      neutral_fallback_threshold=ACK_NEUTRAL_FALLBACK_THRESHOLD),
        emit_fn=tts_emit,
    )
    ack_gate = DeterministicGate()
else:
    ack_coord = None
    ack_gate = None

# Routing certainty handed off from router.route() just before it calls respond().
# Single-threaded request path: route() -> set_route_confidence() -> respond().
_route_confidence = None

# Role-layer outputs for the upcoming generation (cautious path). Set by role_check(),
# consumed once by the next respond(). Single-threaded request path.
_role_directive = None   # wording cue
_role_emotion   = None    # (name, intensity, source, decay_turns) for the emotion layer


def set_route_confidence(value):
    """Called by router.route() to pass NLI routing certainty into generation."""
    global _route_confidence
    _route_confidence = value


def role_check(user_input):
    """Role / identity-preservation gate. Called by router.route() before generation.
    Returns 'proceed' | 'cautious' | 'refuse'. On 'cautious', stashes a wording cue
    and an identity_preservation emotion candidate for the upcoming respond(). On
    'refuse', registers the emotion immediately (no generation will run)."""
    global _role_directive, _role_emotion
    _role_directive = None
    _role_emotion = None
    if role is None:
        return "proceed"
    with _stage("role_check"):
        result = role.evaluate(user_input)
    if result.decision == "refuse":
        if emotion is not None:
            emotion.note("identity_preservation", 0.5, "role_refusal", 2)
            runtime_state.set_flag("last_emotion", emotion.current().primary_state)
        return "refuse"
    if result.decision == "cautious":
        _role_directive = result.cue
        _role_emotion = ("identity_preservation", 0.5, "role_boundary", 2)
        return "cautious"
    return "proceed"


def _apply_emotion(user_input, pre):
    """Run the emotion pre-turn judge and return a wording cue to append to the
    system prompt. Mirrors the primary state into runtime_state for observability.
    Injects any pending role-layer emotion as an external candidate so it competes
    through the normal single-primary lifecycle. Returns "" when disabled/neutral."""
    global _role_emotion
    if emotion is None:
        _role_emotion = None
        return ""
    state = emotion.pre_turn(
        user_input,
        memory_confidence=pre.memory_confidence,
        memory_valid=pre.memory_valid,
        conflict=pre.conflict,
        external=_role_emotion,
    )
    _role_emotion = None   # consumed
    runtime_state.set_flag("last_emotion", state.primary_state)
    return emotion.system_cue()


def _apply_role_directive():
    """Return and consume the pending role-layer wording cue (cautious path)."""
    global _role_directive
    cue = _role_directive or ""
    _role_directive = None
    return cue


def _close_emotion(assistant_text):
    """Run the response-driven emotion judge (e.g. self-limit) after generation."""
    if emotion is None:
        return
    state = emotion.post_turn(assistant_text)
    runtime_state.set_flag("last_emotion", state.primary_state)


def _apply_confidence(user_input, pre):
    """Assess first-class confidence from routing certainty + memory grounding and
    return a wording cue. Consumes the route-confidence handoff (one-shot)."""
    global _route_confidence
    if confidence is None:
        _route_confidence = None
        return ""
    result = confidence.assess(
        _route_confidence, pre.memory_confidence, pre.memory_valid, user_input
    )
    _route_confidence = None   # consumed; router must set it again next turn
    return result.cue


def _fire_ack(user_input):
    """Input-side acknowledgement (Tiered Cognition Doctrine, Phase A). Run the
    cheap deterministic gate on the raw input and, if it fires, SPEAK one ack
    immediately via the priority slot — BEFORE memory/LLM work — so it masks the
    generation latency that follows. One ack per turn. No-op when disabled.

    Returns nothing; the chosen category is stashed on the coordinator and folded
    into the system prompt later via _ack_cue() (the ack is a promise the answer
    must honour)."""
    if ack_coord is None:
        return
    ack_coord.reset_turn()
    verdict = ack_gate.evaluate(user_input)
    ack_coord.submit_gate(verdict)
    ack_coord.resolve_and_emit()   # speaks now if a candidate survived; else silent


def _ack_cue():
    """Return the forward-fed style constraint for the ack emitted this turn ("" if
    none). Folded into the system prompt so Qwen honours what the ack promised."""
    if ack_coord is None:
        return ""
    return ack_coord.prompt_constraint()


_memory          = MemoryManager()
_context_summary = ""
_exchange_count  = 0

# Session-start persona gate (CHARACTER_SESSION_START_ONLY). Resets to False each
# process start, so SYSTEM_PROMPT is emitted on the first turn and then drops out
# of the per-turn system block. clear_history() re-arms it for a fresh conversation.
_session_character_injected = False


# ── persistent history ────────────────────────────────────────────────────────

def _load_history():
    if HISTORY_PATH and os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                print(f"[HISTORY] Loaded {len(data)} messages.", flush=True)
                return data
        except Exception as e:
            print(f"[HISTORY] Load error: {e}", flush=True)
    return []


def _save_history():
    if not HISTORY_PATH:
        return
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(chat_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[HISTORY] Save error: {e}", flush=True)


chat_history = _load_history()


# ── helpers ───────────────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r'(?<=[.!?…])\s+(?=[A-ZÄÖÜ\"\'\(]|$)|(?<=[。！？])')

def _split_sentences(text: str) -> list:
    """Split text into sentences for pipelined TTS — first plays while rest synthesise.
    Short fragments (<10 chars) are merged forward to handle abbreviations like Dr., Nr."""
    if not text:
        return []
    raw = [p.strip() for p in _SENTENCE_END.split(text.strip()) if p.strip()]
    result = []
    i = 0
    while i < len(raw):
        part = raw[i]
        while len(part) < 10 and i + 1 < len(raw):
            i += 1
            part = part + " " + raw[i]
        result.append(part)
        i += 1
    return result


# Sentence terminals that can close a TTS chunk mid-stream.
_TTS_TERMINALS = '.!?…。！？'


def _split_complete_sentences(pending: str):
    """Streaming-TTS helper. Split off sentences that have fully closed and return
    (ready, remainder). The trailing fragment is still growing, so it is held back
    until the next terminal arrives (or the stream ends). Reuses _split_sentences so
    the <10-char abbreviation merge (Dr., Nr.) still applies before anything is emitted."""
    parts = _split_sentences(pending)
    if len(parts) <= 1:
        return [], pending
    return parts[:-1], parts[-1]


def remove_reasoning(text):
    if not text:
        return text
    if SHOW_REASONING:
        return text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


def trim_history(history, max_exchanges):
    return history[-(max_exchanges * 2):]


def _build_system(context_block, summary, cue=""):
    global _session_character_injected
    parts = []
    # Persona: every turn (legacy) OR only the first turn of the session (new).
    if (not CHARACTER_SESSION_START_ONLY) or (not _session_character_injected):
        parts.append(SYSTEM_PROMPT)
        _session_character_injected = True
    if TIME_AWARENESS_ENABLED:
        parts.append(current_time_context())
    if cue:
        parts.append(cue)
    if context_block:
        parts.append(context_block)
    if summary:
        parts.append(f"[Summary of earlier conversation:\n{summary}\n]")
    return "\n\n".join(parts)


def _llm_bare(prompt, max_tok=300):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=max_tok,
        extra_body={"top_k": 1}
    )
    return remove_reasoning(response.choices[0].message.content or "").strip()


# ── memory / summary ──────────────────────────────────────────────────────────

def save_memory(text):
    # User-directed "remember that X" — max salience, never gated.
    if salience is None:
        _memory.add(text)
    else:
        _memory.add(text, metadata={"salience": 1.0, "source": "manual"})
    print(f"[MEMORY] Saved: {text}", flush=True)


def purge_memory():
    global _context_summary
    _memory.purge()
    _context_summary = ""


# Soft cap: let the raw window grow this far before compacting, so compression
# happens in batches (fewer LLM merges => less summary drift) rather than every turn.
_COMPACT_AT_MSGS = (MAX_EXCHANGES + SUMMARY_TRIGGER) * 2

# Guards against concurrent compaction: compaction is stateful (reduces+reassigns
# chat_history/_context_summary), so only one may run at a time or fast successive
# turns would double-fold the same overflow and race the reassignment.
_compact_lock = threading.Lock()


def _compress_overflow(existing_summary, overflow_msgs):
    """Fold the overflow exchanges INTO the existing summary (accumulate, never
    overwrite). Salience-favoured per the Memory/Salience Doctrine: keep identity,
    lasting preferences, projects, relationships, emotionally significant facts;
    drop transient detail. Returns "" on failure so the caller can decline to trim."""
    lines = [
        f"{'User' if m['role'] == 'user' else 'Vivianna'}: {m['content'][:400]}"
        for m in overflow_msgs
    ]
    return _llm_bare(
        "You maintain a running memory summary of an ongoing conversation. "
        "Update it by folding in the new lines below, WITHOUT losing facts already "
        "in the existing summary. Preserve what matters for long-term continuity: "
        "the user's identity, lasting preferences, ongoing projects, relationships, "
        "and emotionally significant facts. Drop transient or trivial detail. "
        "Keep it concise (5-8 sentences). Reply with the updated summary only.\n\n"
        f"Existing summary:\n{existing_summary or '(none yet)'}\n\n"
        "New lines to fold in:\n" + "\n".join(lines),
        max_tok=200,
    )


def _compact_history():
    """Doctrine compression pipeline: when the window exceeds its soft cap, compress
    the overflow (oldest exchanges) into the accumulating summary, THEN trim. Never
    trims before a successful compression — if compression fails, keep the full
    window and retry next turn rather than discarding un-compressed content.

    The compression LLM call runs in a daemon thread (mirrors _auto_save_memory) so
    the main response path returns immediately instead of blocking — this is what
    removes the per-turn CPU spike. _compact_lock ensures only one compaction is in
    flight; if one is already running this turn is skipped and retried next turn. The
    prompt window is always sliced to the last MAX_EXCHANGES*2 msgs at build time, so
    an un-trimmed window between turns never bloats the prompt."""
    global chat_history, _context_summary
    keep = MAX_EXCHANGES * 2
    if len(chat_history) <= _COMPACT_AT_MSGS:
        return
    if not _compact_lock.acquire(blocking=False):
        return                      # compaction already in flight; retry next turn
    overflow = chat_history[:-keep]
    if not overflow:
        _compact_lock.release()
        return

    def _worker():
        global chat_history, _context_summary
        try:
            new_summary = _compress_overflow(_context_summary, overflow)
            if not new_summary:
                print("[MEMORY] Compaction deferred — empty summary result; "
                      "window kept intact.", flush=True)
                return
            _context_summary = new_summary
            chat_history = chat_history[-keep:]
            _save_history()         # persist the trimmed window from the thread
            print(f"[MEMORY] Compacted {len(overflow)} msgs into summary; "
                  f"window now {len(chat_history)} msgs.", flush=True)
        finally:
            _compact_lock.release()

    threading.Thread(target=_worker, daemon=True).start()


def _auto_save_memory(user_input, assistant_text):
    if not PROACTIVE_MEMORY:
        return

    def _worker():
        result = _llm_bare(
            "Does this exchange contain a personal fact, strong preference, or lasting detail about the user "
            "worth remembering permanently? If yes, state it as one clear sentence. "
            "If no, reply exactly: none\n\n"
            f"User: {user_input[:400]}\n"
            f"Vivianna: {assistant_text[:400]}"
        )
        if result and result.lower().strip() not in ("none", "no", ""):
            if salience is None:
                _memory.add(result)
                print(f"[MEMORY] Auto-saved: {result}", flush=True)
            else:
                sr = salience.score(result)
                if sr.score >= SALIENCE_STORE_THRESHOLD:
                    _memory.add(result, metadata={"salience": sr.score, "source": "auto"})
                    print(f"[MEMORY] Auto-saved (salience={sr.score:.2f}): {result}", flush=True)
                else:
                    print(f"[MEMORY] Skipped low salience ({sr.score:.2f} < "
                          f"{SALIENCE_STORE_THRESHOLD}): {result}", flush=True)

    threading.Thread(target=_worker, daemon=True).start()


# ── public API ────────────────────────────────────────────────────────────────

def clear_history():
    global chat_history, _context_summary, _exchange_count, _session_character_injected
    chat_history = []
    _context_summary = ""
    _exchange_count = 0
    _session_character_injected = False   # re-anchor persona once on the next turn
    _save_history()


def print_debug():
    print("\n--- DEBUG ---")
    if len(chat_history) >= 2:
        print(f"LAST USER:      {chat_history[-2]['content'][:300]}")
        print(f"LAST ASSISTANT: {chat_history[-1]['content'][:300]}")
    else:
        print("LAST EXCHANGE: none yet")
    print(f"HISTORY SIZE: {len(chat_history)}")
    print(f"MEMORIES:     {_memory.count()}")
    print(f"SUMMARY:      {_context_summary[:200] if _context_summary else 'none'}")
    print(f"EMOTION:      {emotion.current().as_dict() if emotion else 'disabled'}")
    if confidence:
        _c = confidence.last
        print(f"CONFIDENCE:   score={_c.score:.2f} band={_c.band} [{_c.source}]")
    else:
        print("CONFIDENCE:   disabled")
    print("----------------\n")


def nli_classify(user_input):
    tool_lines = "\n".join(
        f"- {t.name}: {t.description}"
        for t in sorted(TOOLS, key=lambda x: x.priority)
    )
    with _stage("nli_classify"):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": (
                "Answer in English only. You are a classifier.\n"
                "Select the best tool for this message and your confidence (0.0-1.0).\n\n"
                f"Tools:\n{tool_lines}\n\n"
                "Reply in exactly this format: tool_name confidence\n"
                "Example: web 0.92\n\n"
                f"Message: {user_input}"
            )}],
            temperature=0.1,
            max_tokens=10,
            extra_body={"top_k": 1}
        )
    raw = remove_reasoning(response.choices[0].message.content or "").strip().lower()
    parts = raw.split()
    tool_name = parts[0] if parts else "chat"
    try:
        confidence = float(parts[1]) if len(parts) > 1 else 0.5
    except ValueError:
        confidence = 0.5
    print(f"[NLI] tool={tool_name} confidence={confidence:.2f} raw={raw!r}", flush=True)
    # Calibration trace: greppable single line (filter with `findstr [NLI-CALIB]`) to
    # eyeball the confidence distribution on real queries BEFORE trusting the three-tier
    # bands (WEB_CONFIDENCE_HIGH/MEDIUM/LOW). Watch for bimodal collapse (~0.5 / ~0.9).
    print(f"[NLI-CALIB]\tconf={confidence:.2f}\ttool={tool_name}\tq={user_input!r}", flush=True)
    return tool_name, confidence


def llm_plain(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt}
        ],
        temperature=0.3,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        extra_body={"top_k": TOP_K, "min_p": MIN_P,
                    "repeat_penalty": REPEAT_PENALTY, "dry_multiplier": DRY_MULTIPLIER}
    )
    text = remove_reasoning(response.choices[0].message.content)
    return text.strip() if text else "[No response received]"


# ── respond ───────────────────────────────────────────────────────────────────

def respond_streaming(user_input, llm_input=None, tool_driven=False):
    global chat_history, _exchange_count

    # llm_input: callers (e.g. web search) send augmented content to the model while
    # only the original user_input is stored in history — prevents web-result bloat.
    # tool_driven: deterministic tool answers (web search) are barred from long-term memory.
    llm_text = llm_input if llm_input is not None else user_input

    runtime_state.set_flag("thinking", True)

    # Input-side ack FIRST: speak it now so it plays during the work below.
    with _stage("fire_ack"):
        _fire_ack(user_input)

    with _stage("memory.query"):
        mem_hits = _memory.query(user_input)
    with _stage("pre_generate"):
        pre = stabilizer.pre_generate(user_input, mem_hits)
    role_cue = _apply_role_directive()
    with _stage("apply_emotion"):
        emo_cue = _apply_emotion(user_input, pre)
    with _stage("apply_confidence"):
        conf_cue = _apply_confidence(user_input, pre)
    with _stage("memory.context_block"):
        context_block = _memory.context_block(user_input)
    ack_cue       = _ack_cue()
    cue           = " ".join(p for p in (pre.system_cue, role_cue, emo_cue, conf_cue, ack_cue) if p)
    sys_prompt    = _build_system(context_block, _context_summary, cue=cue)

    messages = [
        {"role": "system", "content": sys_prompt},
        *chat_history[-(MAX_EXCHANGES * 2):],
        {"role": "user",   "content": llm_text}
    ]

    with _stage("create(stream) handshake"):
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            top_p=TOP_P,
            frequency_penalty=FREQUENCY_PENALTY,
            presence_penalty=PRESENCE_PENALTY,
            stream=True,
            extra_body={"top_k": TOP_K, "min_p": MIN_P,
                        "repeat_penalty": REPEAT_PENALTY, "dry_multiplier": DRY_MULTIPLIER}
        )

    runtime_state.set_flag("thinking", False)
    runtime_state.set_flag("speaking", True)

    print("\nVivianna: ", end="", flush=True)

    FLUSH_ON = {'.', ',', '!', '?', ';', ':', '…', '。', '！', '？'}

    display_buf      = []
    tts_pending      = ""      # raw text awaiting a sentence boundary; handed to TTS incrementally
    full_text        = ""
    t_start          = time.perf_counter()
    first_token_time = None
    stream_start     = None

    # Pipeline: LLM stream → stabilizer.iter_stream() → console + incremental TTS queue.
    # Each sentence is handed to TTS the moment it closes, so sentence 1 synthesises while
    # later sentences are still being generated (instead of waiting for the whole reply).
    for event in stabilizer.iter_stream(stream):
        if event.suppressed:
            continue
        delta = event.text
        if not delta:
            continue

        if first_token_time is None:
            first_token_time = time.perf_counter()
            print(f"[TTFT: {first_token_time - t_start:.2f}s]", flush=True)

        full_text   += delta
        display_buf.append(delta)
        tts_pending += delta

        # Console display: flush on punctuation boundaries
        if any(c in delta for c in FLUSH_ON):
            if stream_start is None:
                stream_start = time.perf_counter()
                print(f"[STREAM START: {stream_start - t_start:.2f}s]", flush=True)
            print("".join(display_buf), end="", flush=True)
            display_buf = []

        # TTS: emit every sentence that has fully closed; keep the in-progress tail buffered
        if any(c in delta for c in _TTS_TERMINALS):
            ready, tts_pending = _split_complete_sentences(tts_pending)
            for sentence in ready:
                cleaned = clean_for_tts(sentence)
                if cleaned:
                    tts_emit(cleaned, source="llm", display=False)

    # Flush remaining display buffer
    if display_buf:
        if stream_start is None:
            print(f"[STREAM START: {time.perf_counter() - t_start:.2f}s]", flush=True)
        print("".join(display_buf), end="", flush=True)

    # Flush the final (in-progress) sentence to TTS
    tail = clean_for_tts(tts_pending)
    if tail:
        tts_emit(tail, source="llm", display=False)

    print()
    runtime_state.set_flag("speaking", False)

    full_text = remove_reasoning(full_text)
    if not full_text:
        full_text = t('no_response', user_input)

    post = stabilizer.post_generate(user_input, full_text, pre, tool_driven=tool_driven)
    _close_emotion(post.clean_text)

    chat_history.append({"role": "user",      "content": user_input})
    chat_history.append({"role": "assistant", "content": post.clean_text})
    _exchange_count += 1
    _compact_history()          # doctrine: compress overflow into summary, THEN trim
    _save_history()

    if post.commit_to_memory:
        _auto_save_memory(user_input, post.clean_text)

    return None


def respond(user_input, llm_input=None, tool_driven=False):
    global chat_history, _exchange_count

    # llm_input: see respond_streaming — keeps web-augmented text out of history.
    # tool_driven: see respond_streaming — bars deterministic tool answers from memory.
    llm_text = llm_input if llm_input is not None else user_input

    _fire_ack(user_input)   # input-side ack: speak now, before generation
    pre           = stabilizer.pre_generate(user_input, _memory.query(user_input))
    role_cue      = _apply_role_directive()
    emo_cue       = _apply_emotion(user_input, pre)
    conf_cue      = _apply_confidence(user_input, pre)
    context_block = _memory.context_block(user_input)
    ack_cue       = _ack_cue()
    cue           = " ".join(p for p in (pre.system_cue, role_cue, emo_cue, conf_cue, ack_cue) if p)
    sys_prompt    = _build_system(context_block, _context_summary, cue=cue)

    messages = [
        {"role": "system", "content": sys_prompt},
        *chat_history[-(MAX_EXCHANGES * 2):],
        {"role": "user",   "content": llm_text}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        extra_body={"top_k": TOP_K, "min_p": MIN_P,
                    "repeat_penalty": REPEAT_PENALTY, "dry_multiplier": DRY_MULTIPLIER}
    )

    assistant_text = remove_reasoning(response.choices[0].message.content)
    if not assistant_text:
        assistant_text = "[No response received]"

    post = stabilizer.post_generate(user_input, assistant_text, pre, tool_driven=tool_driven)
    _close_emotion(post.clean_text)

    chat_history.append({"role": "user",      "content": user_input})
    chat_history.append({"role": "assistant", "content": post.clean_text})
    _exchange_count += 1
    _compact_history()          # doctrine: compress overflow into summary, THEN trim
    _save_history()

    if post.commit_to_memory:
        _auto_save_memory(user_input, post.clean_text)

    return post.clean_text
