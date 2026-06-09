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
    EMOTION_MODEL_ENABLED, EMOTION_MODEL_NAME, EMOTION_MODEL_DEVICE,
    EMOTION_MODEL_THRESHOLD, EMOTION_MODEL_INTENSITY_SCALE,
    SALIENCE_LAYER_ENABLED, SALIENCE_DEBUG, SALIENCE_STORE_THRESHOLD,
    SALIENCE_MODEL_ENABLED, SALIENCE_MODEL_NAME, SALIENCE_MODEL_DEVICE,
    SALIENCE_BACKEND, SALIENCE_AGG, SALIENCE_AGG_TEMP, CROSS_ENCODER_MODEL,
    CROSS_ENCODER_DEVICE, MEMORY_RERANK_ENABLED,
    GROUNDING_ENABLED, GROUNDING_MODEL, GROUNDING_DEVICE, GROUNDING_CONTRADICT_THRESHOLD,
    WEB_GROUNDING_CHECK_ENABLED, WEB_GROUNDING_ENTAIL_FLOOR, WEB_GROUNDING_MIN_SENT_CHARS,
    WEB_GROUNDING_DISCLAIMER_MARKERS,
    CLARIFY_ENABLED, CLARIFY_MIN_AGE_HOURS,
    CONFIDENCE_LAYER_ENABLED, CONFIDENCE_DEBUG,
    ROLE_LAYER_ENABLED, ROLE_DEBUG,
    ACK_LAYER_ENABLED, ACK_DEBUG, ACK_MODE, ACK_NEUTRAL_FALLBACK_THRESHOLD,
    TIME_AWARENESS_ENABLED,
    CHARACTER_SESSION_START_ONLY,
)
from tool_registry import TOOLS
from commands import RECALL_LTM, RECALL_CONTEXT, RECALL_COMBINED
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
stabilizer = ViviannaStabilizer(StabilizerConfig(
    grounding_enabled=GROUNDING_ENABLED,
    grounding_contradict_threshold=GROUNDING_CONTRADICT_THRESHOLD,
))
emotion    = EmotionLayer(EmotionConfig(
    debug=EMOTION_DEBUG,
    go_emotions_threshold=EMOTION_MODEL_THRESHOLD,
    go_emotions_intensity_scale=EMOTION_MODEL_INTENSITY_SCALE,
)) if EMOTION_LAYER_ENABLED else None
salience   = SalienceLayer(SalienceConfig(
    debug=SALIENCE_DEBUG,
    use_model=SALIENCE_MODEL_ENABLED,
    backend=SALIENCE_BACKEND,
    model_name=SALIENCE_MODEL_NAME,
    xenc_model=CROSS_ENCODER_MODEL,
    agg=SALIENCE_AGG,
    agg_temp=SALIENCE_AGG_TEMP,
    model_device=SALIENCE_MODEL_DEVICE,
)) if SALIENCE_LAYER_ENABLED else None
# Pre-warm the shared cross-encoder during startup so its ~6s cold-load happens
# while the user reads the boot lines, not on their first memory turn. Only when
# something actually uses it (salience cross-encoder backend or memory rerank).
if SALIENCE_BACKEND == "cross-encoder" or MEMORY_RERANK_ENABLED:
    import cross_encoder_model
    cross_encoder_model.prewarm(CROSS_ENCODER_MODEL, CROSS_ENCODER_DEVICE)
# Pre-warm the grounding/contradiction NLI (encoder-stack-v2 third piece) the same way, so
# its cold-load overlaps the boot lines, not the first memory turn. Gated on GROUNDING_ENABLED.
# Loads ~301MB fp16 on CUDA; a load failure inside leaves consumers (3c/3d) on their fallback
# (cosine-only validity / no contradiction check). No conversation behavior changes until 3c/3d.
if GROUNDING_ENABLED:
    import grounding
    grounding.prewarm(GROUNDING_MODEL, GROUNDING_DEVICE)
# Pre-warm the go_emotions affective-tone judge (encoder-stack-v2) the same way, so its CPU
# cold-load overlaps the boot lines, not the user's first turn. Gated on both the layer and the
# model flag. Load failure inside -> emotion falls back to regex-only (no behavior change).
if EMOTION_LAYER_ENABLED and EMOTION_MODEL_ENABLED:
    import emotion_model
    emotion_model.prewarm(EMOTION_MODEL_NAME, EMOTION_MODEL_DEVICE)
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
_role_directive = None   # wording cue (identity is owned solely by RoleLayer; see role_check)


def set_route_confidence(value):
    """Called by router.route() to pass NLI routing certainty into generation."""
    global _route_confidence
    _route_confidence = value


def role_check(user_input):
    """Role / identity-preservation gate. Called by router.route() before generation.
    Returns 'proceed' | 'cautious' | 'refuse'. On 'cautious', stashes a wording cue
    (role_directive) for the upcoming respond(). Identity is owned entirely by this
    layer and delivered via the independent `role_cue` prompt channel — it is NOT
    routed through the emotion layer's single-primary contest (which now belongs to
    the affective tone judge alone)."""
    global _role_directive
    _role_directive = None
    if role is None:
        return "proceed"
    with _stage("role_check"):
        result = role.evaluate(user_input)
    if result.decision == "refuse":
        return "refuse"
    if result.decision == "cautious":
        _role_directive = result.cue
        return "cautious"
    return "proceed"


def _apply_emotion(user_input, pre):
    """Run the emotion pre-turn judge and return a wording cue to append to the
    system prompt. Mirrors the primary state into runtime_state for observability.
    Identity preservation is no longer injected here — it rides the independent
    role_cue channel (see role_check). Returns "" when disabled/neutral."""
    if emotion is None:
        return ""
    # Semantic affective read of the USER message (go_emotions). Fallback-safe: classify returns
    # None on any failure/disabled -> pre_turn behaves regex-only. One CPU pass (~15ms), masked
    # by the input ack + generation handshake.
    model_scores = None
    if EMOTION_MODEL_ENABLED:
        try:
            import emotion_model
            model_scores = emotion_model.classify(user_input)
        except Exception:  # noqa: BLE001 — never let the tone read break a turn
            model_scores = None
    state = emotion.pre_turn(
        user_input,
        memory_confidence=pre.memory_confidence,
        memory_valid=pre.memory_valid,
        conflict=pre.conflict,
        model_scores=model_scores,
    )
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


def last_user_message():
    """The most recent USER turn already in history. At router time the CURRENT input has not
    been appended yet (that happens after generation), so this returns the PRIOR user turn —
    the topic a referential follow-up ('look it up') really refers to. '' if none. Used by
    router._resolve_search_query (item 2 query repair)."""
    for m in reversed(chat_history):
        if m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


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


# ── memory audit (the /audit command) ─────────────────────────────────────────
def _parse_audit(out):
    """Parse the LLM cluster judgement into (verdict, merged_text)."""
    verdict, merged = "REVIEW", ""
    for line in (out or "").splitlines():
        s = line.strip()
        up = s.upper()
        if up.startswith("VERDICT:"):
            v = up.split(":", 1)[1]
            for k in ("DUPLICATE", "CONFLICT", "DISTINCT"):
                if k in v:
                    verdict = k
                    break
        elif up.startswith("MERGED:"):
            merged = s.split(":", 1)[1].strip()
    if merged in ("-", "—", "none", "None"):
        merged = ""
    return verdict, merged


def audit_scan(cos_threshold=None):
    """Audit the memory store for redundancy/conflict. NO side effects.

    Cosine-clusters near-duplicate entries (cheap, local), then asks the LLM to
    judge each cluster as DUPLICATE (-> one merged third-person sentence), CONFLICT,
    or DISTINCT. Returns (proposals, entries) where each proposal is
    {indices, texts, verdict, merged}. Needs llama-server for the judgement; an
    unparseable reply yields verdict='REVIEW' (never auto-applied)."""
    from config import MEMORY_AUDIT_CLUSTER_COS
    th = MEMORY_AUDIT_CLUSTER_COS if cos_threshold is None else cos_threshold
    entries = _memory.all_entries()
    clusters = _memory.cluster_candidates(th)
    proposals = []
    for cl in clusters:
        texts = [entries[i]["text"] for i in cl]
        numbered = "\n".join(f"{n + 1}. {tx}" for n, tx in enumerate(texts))
        out = _llm_bare(
            "These long-term memory notes about the user may overlap. Classify them:\n"
            "- DUPLICATE: they describe the SAME person and facts; one may reword the other or add "
            "extra detail. Extra or missing details do NOT make them conflict — they should be MERGED.\n"
            "- CONFLICT: a specific fact in one DIRECTLY contradicts another (e.g. 'allergic to "
            "penicillin' vs 'not allergic to penicillin'; two different jobs for the same person).\n"
            "- DISTINCT: they are unrelated separate facts.\n"
            "Reply in EXACTLY this format, two lines:\n"
            "VERDICT: <DUPLICATE|CONFLICT|DISTINCT>\n"
            "MERGED: <if DUPLICATE, ONE third-person sentence beginning with 'The user' that combines "
            "EVERY detail from all the notes and loses nothing; otherwise a single dash>\n\n"
            f"Notes:\n{numbered}"
        )
        verdict, merged = _parse_audit(out)
        proposals.append({"indices": cl, "texts": texts,
                          "verdict": verdict, "merged": merged})
    return proposals, entries


def apply_audit(merges=None, deletes=None):
    """Apply approved audit actions in one index-safe batch.
    merges: [(indices, merged_text), ...]; deletes: flat index list. Returns count."""
    return _memory.apply_audit_batch(merges or [], deletes or [])


def draft_merge(texts):
    """Draft ONE third-person sentence combining an arbitrary set of memory notes,
    for the manual CONFLICT resolver in /audit (where the user hand-picks which
    entries to merge rather than taking a whole-cluster verdict). Returns the
    sentence, or '' if the LLM is unreachable / replies empty."""
    numbered = "\n".join(f"{n + 1}. {tx}" for n, tx in enumerate(texts))
    out = _llm_bare(
        "Combine the following long-term memory notes about the user into ONE "
        "third-person sentence beginning with 'The user' that keeps EVERY detail "
        "and loses nothing. Reply with only that sentence.\n\n"
        f"Notes:\n{numbered}"
    )
    return (out or "").strip()


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


def _grounding_contradicts(candidate):
    """3d write-time contradiction guard. Returns the text of a nearby EXISTING memory that
    the new `candidate` fact CONTRADICTS (NLI premise=existing, claim=candidate), else None.

    Why this exists alongside 3c: 3c (stabilizer.pre_generate) only sees the turn's retrieved
    top-k, so it blocks a save only when the user's CURRENT message contradicts a RETRIEVED
    memory. A candidate that contradicts a stored memory NOT retrieved this turn slips past 3c.
    3d re-queries the store for the candidate's nearest neighbors (a contradiction is
    necessarily topically related -> shows up in top-k, so no all-pairs scan) and checks those.

    Policy (Jamie 2026-06-07): block + log, never auto-overwrite. Fallback-safe: grounding
    import fail / model unavailable -> None -> behave exactly as before (no guard)."""
    if not GROUNDING_ENABLED:
        return None
    try:
        import grounding
    except Exception:
        return None
    thr = GROUNDING_CONTRADICT_THRESHOLD
    for n in _memory.query(candidate):
        ev = (n.get("text") if isinstance(n, dict) else getattr(n, "text", "")) or ""
        if not ev:
            continue
        p = grounding.contradiction_prob(ev, candidate)   # premise=existing, claim=candidate
        if p is None:            # model unavailable -> behave as before
            return None
        if p >= thr:
            return ev
    return None


# Split on sentence-final punctuation followed by whitespace. Good enough for grounding a
# spoken-style answer into claim-sized units; not a full NLP segmenter.
_WEB_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ── [new-3] search self-correction: SUGGEST_SEARCH marker ─────────────────────
# When a web/source answer can't be found in the page, the honesty prompt asks the model to
# append a machine-readable final line `SUGGEST_SEARCH: <better query>`. We parse that query
# out (to arm a re-search offer) and keep it OUT of the spoken/displayed answer.
_SUGGEST_MARK = "SUGGEST_SEARCH:"


def _suppress_suggest(full_text):
    """Streaming predicate: True once the accumulated text has reached (or begun) the
    SUGGEST_SEARCH line, so the stream loop stops emitting it to console/TTS. Checks the tail
    after the last newline against a growing prefix of the marker, so token fragments
    ('SUG' -> 'SUGGEST' -> ...) are caught before any of it leaks. Requires >=3 chars to avoid
    suppressing on a lone 'S'."""
    if _SUGGEST_MARK in full_text:
        return True
    tail = full_text.rsplit("\n", 1)[-1].lstrip()
    return len(tail) >= 3 and _SUGGEST_MARK.startswith(tail)


def _extract_suggestion(full_text):
    """Split a finished reply into (clean_answer, suggested_query_or_None). The suggestion is the
    first line after the marker; everything from the marker on is removed from the answer."""
    i = full_text.find(_SUGGEST_MARK)
    if i == -1:
        return full_text.rstrip(), None
    answer = full_text[:i].rstrip()
    rest = full_text[i + len(_SUGGEST_MARK):].strip()
    # Strip wrapping quotes the model often adds — a quoted DDG query forces an exact-phrase
    # match (live run 5: '"Jay Chou biography..."' → tangential 2015 results).
    suggested = rest.splitlines()[0].strip().strip('"').strip("'").strip() if rest else ""
    return answer, (suggested or None)


# [new-3] B2: a parked re-search offer. PARALLEL to _pending_clarify — never shares that state,
# so the two pending-question types can't collide. respond_streaming arms it from a parsed
# SUGGEST_SEARCH; main.py resolves the next-turn yes/no via router.resolve_research, which runs
# the query back through the web path. Loop cap: the executed re-search turn passes
# allow_research_arm=False so a second fail can't re-arm (no fail->suggest->fail spiral).
_pending_research = None   # None or {"query": str}


def pending_research():
    """The parked re-search offer (or None) so main.py can resolve the user's next-turn yes/no."""
    return _pending_research


def clear_pending_research():
    global _pending_research
    _pending_research = None


def _research_offer_text(query):
    """The SPOKEN offer for a re-search. Deterministic + always shows the exact query we'd run
    (curtain-test: the user sees what 'yes' will do)."""
    return f'I couldn\'t find it there — want me to try searching for "{query}" instead?'


def _arm_research(query):
    """Park the re-search offer AND speak it. Arming and asking are ONE act: the model is told
    NOT to ask, so this is the only place the offer is made — otherwise the handler arms in
    silence and only a debug-watcher could answer it (live run 5 bug)."""
    global _pending_research
    _pending_research = {"query": query}
    print(f"[RESEARCH] armed re-search offer: {query[:70]!r}", flush=True)
    tts_emit(_research_offer_text(query), source="clarification")


def _is_web_disclaimer(sentence):
    """True if the sentence is an HONEST meta/disclaimer — it says the answer is NOT in the page
    or that Vivianna couldn't/can't find it (5a). Such a sentence is correctly NOT entailed by
    the page, so it must be skipped, not flagged as confabulation. Markers are tight (see
    WEB_GROUNDING_DISCLAIMER_MARKERS) to avoid swallowing a real confab that merely opens 'I…'."""
    low = sentence.lower()
    return any(m in low for m in WEB_GROUNDING_DISCLAIMER_MARKERS)


def _web_answer_grounding(evidence, answer):
    """LOG-ONLY web-answer grounding guard (answer-precision). For each FACTUAL sentence in the
    generated answer, score NLI entailment by the fetched page (premise=page, hypothesis=
    sentence) and log it; report the weakest-supported sentence — the most likely confabulation.

    Source selection (the two-stage rerank) picks the right page; this guards the answer the
    model writes over it. Motivated live 2026-06-08: correct wedding article picked, but a
    fabricated date+venue the page never contained slipped through (a domain-expert listener
    caught it, not the system).

    NEVER suppresses or hedges (yet) — it only emits [WEB-GROUND] entailment data to calibrate
    WEB_GROUNDING_ENTAIL_FLOOR before that floor is ever allowed to gate an answer (same
    discipline as WEB_RELEVANCE_FLOOR; honours the memory-governance 'detect, don't mutate
    without authority' line). Fallback-safe: disabled / no evidence / model unavailable / any
    error -> silent no-op, answer unchanged. Per-sentence passes are sequential, so peak VRAM
    matches a single 3c/3d grounding pass (premise length, not pass count, drives the spike)."""
    if not (GROUNDING_ENABLED and WEB_GROUNDING_CHECK_ENABLED):
        return
    if not evidence or not answer:
        return
    try:
        import grounding
        sents = [s.strip() for s in _WEB_SENT_SPLIT.split(answer.strip()) if s.strip()]
        # Factual claims only: a trailing question ("Want to know more?") or short filler is not
        # entailed by anything and would pollute the minimum, masking the real weakest claim.
        # 5a: also skip honest disclaimer/meta sentences ("I couldn't find X", "the search
        # results don't contain Y") — they are correctly un-entailed by the page and would be
        # false-flagged as confabulation (the very thing this guard is meant to catch).
        claims = []
        for s in sents:
            if len(s) < WEB_GROUNDING_MIN_SENT_CHARS or s.rstrip().endswith("?"):
                continue
            if _is_web_disclaimer(s):
                print(f"[WEB-GROUND] skip (disclaimer)  {s[:80]}", flush=True)
                continue
            claims.append(s)
        if not claims:
            return
        worst_p, worst_s = 1.0, ""
        for s in claims:
            p = grounding.entailment_prob(evidence, s)   # premise=page, claim=sentence
            if p is None:        # model unavailable -> behave exactly as before (no guard)
                return
            tag = " LOW" if p < WEB_GROUNDING_ENTAIL_FLOOR else ""
            print(f"[WEB-GROUND] entail={p:.3f}{tag}  {s[:80]}", flush=True)
            if p < worst_p:
                worst_p, worst_s = p, s
        flag = "LIKELY-CONFABULATION" if worst_p < WEB_GROUNDING_ENTAIL_FLOOR else "ok"
        print(f"[WEB-GROUND] min_entail={worst_p:.3f} [{flag}] weakest={worst_s[:80]!r}",
              flush=True)
    except Exception as e:  # noqa: BLE001 — a log-only guard must never break a turn
        print(f"[WEB-GROUND] check skipped ({type(e).__name__}: {e}).", flush=True)


def _auto_save_memory(user_input, assistant_text):
    if not PROACTIVE_MEMORY:
        return

    def _worker():
        # Extraction framing, NOT a yes/no question: the old "does this contain a
        # fact worth remembering permanently? else reply 'none'" phrasing made the
        # 9B over-anchor on the 'none' escape hatch and reject ~5/6 real facts under
        # greedy decoding (name/occupation/family/preference all dropped). Reframing
        # as "extract any lasting fact ... else none" recovered 11/11 (5 facts kept,
        # 6 transient/greeting/question cases still rejected). See instant-rollback
        # backup for the original prompt.
        #
        # USER MESSAGE ONLY (assistant_text intentionally NOT fed in): on a recall /
        # audit turn the user states no new fact, but Vivianna's reply recites every
        # stored fact — feeding that reply to the extractor made it manufacture a
        # superset sentence and re-store it (the over-store seen live during /audit,
        # e.g. "Jamie is a freelance fashion designer with blood type O who loves
        # fresh fruit."). Facts must come from what the USER says, not from Vivianna
        # reciting memory back, so the extractor only ever reads the user's own words.
        #
        # THIRD PERSON ("The user ...") is enforced for two reasons: (1) it matches the
        # other write paths (draft_merge / audit merge) and the third-person distribution
        # the salience (0.97) and rerank thresholds were tuned on; (2) it keeps the dedup
        # guard (memory.add cosine>=0.95) working — a 2nd-person "You are X" vs stored
        # "The user is X" embed far enough apart to slip past dedup and accumulate.
        # Owner identity is NOT put in the sentence text: the multi-user design (Model A)
        # carries user_id in metadata and filters query/dedup by active profile, so the
        # sentence stays generic. (Profile-scoped dedup/query + speaker id into this fn
        # are still TODO, gated on the profile-switch lead.)
        result = _llm_bare(
            "You maintain long-term memory about the user. From the user's message below, extract any lasting "
            "fact the user states about themselves (identity, family, health, work, stable preference) as one "
            "short sentence in the third person, beginning with 'The user'. If the message states no lasting "
            "fact (a greeting, question, request, or small talk), reply exactly: none\n\n"
            f"User: {user_input[:400]}"
        )
        if result and result.lower().strip() not in ("none", "no", ""):
            # 3d write-time contradiction guard: never silently store a fact that contradicts
            # an existing memory. Block + log (Jamie's policy); resolution is via restate or
            # /audit, not auto-overwrite. Catches contradictions 3c missed (non-retrieved mem).
            conflict_ev = _grounding_contradicts(result)
            if conflict_ev is not None:
                print(f"[MEMORY] Contradiction with existing ('{conflict_ev[:60]}'); "
                      f"not auto-saved: {result}", flush=True)
                return
            # dedup=True: a recall turn re-extracting an already-stored fact is a
            # near-duplicate; memory.add skips it (and logs) instead of accumulating.
            if salience is None:
                if _memory.add(result, dedup=True):
                    print(f"[MEMORY] Auto-saved: {result}", flush=True)
            else:
                sr = salience.score(result)
                if sr.score >= SALIENCE_STORE_THRESHOLD:
                    if _memory.add(result, metadata={"salience": sr.score, "source": "auto"},
                                   dedup=True):
                        print(f"[MEMORY] Auto-saved (salience={sr.score:.2f}): {result}", flush=True)
                    # else: near-duplicate; memory.add already logged the dedup skip
                else:
                    print(f"[MEMORY] Skipped low salience ({sr.score:.2f} < "
                          f"{SALIENCE_STORE_THRESHOLD}): {result}", flush=True)

    threading.Thread(target=_worker, daemon=True).start()


# ── time-gated contradiction clarify-and-resolve (vivianna_contradiction_clarify_plan) ──
# Extends 3c: when the user's message contradicts an OLD retrieved memory
# (>= CLARIFY_MIN_AGE_HOURS), Vivianna ASKS at the end of her reply whether it changed; the
# next turn's answer is resolved here into update / delete / keep+re-stamp. A contradiction vs
# a FRESH memory stays pure-3c (honesty cue only — likely a mood/joke, not a life change).
# Cross-turn state mirrors main._pending_recall: the question is cue-injected into the normal
# reply (decision 4); resolution runs at the top of the main loop via resolve_clarify().
_pending_clarify = None   # None, or {"memory_text": str, "user_claim": str}


def pending_clarify():
    """The parked clarify dict (or None) so main.py can resolve the user's next-turn answer."""
    return _pending_clarify


def clear_pending_clarify():
    global _pending_clarify
    _pending_clarify = None


def _effective_age_hours(metadata):
    """Hours since a memory was last established true: confirmed_at if it was re-validated,
    else created_at. None when neither is stamped (legacy entries) -> caller treats as old."""
    md = metadata or {}
    ts = md.get("confirmed_at")
    if not isinstance(ts, (int, float)):
        ts = md.get("created_at")
    if not isinstance(ts, (int, float)):
        return None
    return max(0.0, (time.time() - ts) / 3600.0)


def _maybe_arm_clarify(user_input, mem_hits, pre):
    """If 3c flagged a contradiction against an OLD memory, park it and return a cue telling
    Vivianna to ask (at the end, in persona) whether it changed. Returns "" when disabled, no
    contradiction, or every contradicted memory is too FRESH to be more than noise (those keep
    the existing 3c honesty-cue behavior). Picks the OLDEST eligible contradicted memory. Runs
    in the main respond thread (synchronous) so _pending_clarify is single-writer."""
    global _pending_clarify
    if not CLARIFY_ENABLED or not pre.conflict_texts:
        return ""
    # Map each contradicted text back to its stored metadata (for age) via this turn's hits.
    by_text = {}
    for h in mem_hits:
        txt = (h.get("text") if isinstance(h, dict) else getattr(h, "text", "")) or ""
        if txt:
            by_text.setdefault(txt, (h.get("metadata") if isinstance(h, dict) else None) or {})
    best_text, best_rank = None, None
    for txt in pre.conflict_texts:
        age = _effective_age_hours(by_text.get(txt, {}))
        if age is not None and age < CLARIFY_MIN_AGE_HOURS:
            continue                                   # fresh -> noise, leave to 3c
        rank = float("inf") if age is None else age    # unknown/legacy ranks oldest
        if best_rank is None or rank > best_rank:
            best_text, best_rank = txt, rank
    if best_text is None:
        print(f"[CLARIFY] contradiction(s) but all fresh (<{CLARIFY_MIN_AGE_HOURS}h) — 3c only.",
              flush=True)
        return ""
    _pending_clarify = {"memory_text": best_text, "user_claim": user_input}
    age_str = "unknown age" if best_rank == float("inf") else f"~{best_rank:.0f}h old"
    print(f"[CLARIFY] armed on contradicted memory ({age_str}): {best_text[:60]}", flush=True)
    return (
        f'Separately, a while ago the user told you: "{best_text}". What they just said seems '
        "to contradict that. After answering them naturally, gently and briefly ask — in your "
        "own voice as Vivianna — whether that has changed. Do not assume it has; just check."
    )


def _classify_clarify_answer(user_input, old_text, user_claim):
    """Interpret the user's reply to a clarify question. Returns (action, new_text): action in
    {KEEP, DELETE, UPDATE, HOLD, UNRELATED}; new_text is the 3rd-person replacement fact for
    UPDATE else None. The NEW value lives in user_claim (the statement that triggered the clarify
    last turn), NOT in user_input (the bare confirmation, e.g. "yes please do so" — its referent
    is the change Vivianna offered, which is unrecoverable from the reply alone). Both are given
    to the model so an affirmation resolves to UPDATE carrying the right value. Greedy _llm_bare,
    mirroring _auto_save_memory's reliable framing."""
    out = _llm_bare(
        "A fact you have stored about the user seems to be contradicted by something they said, "
        "so you asked them whether it has really changed. Decide what to do with the stored fact.\n\n"
        f'Stored fact: "{old_text}"\n'
        f'What the user said that prompted the question (the possible new value): "{(user_claim or "")[:400]}"\n'
        f'Their reply to your question: "{user_input[:400]}"\n\n'
        "Reply with EXACTLY one of these, nothing else:\n"
        "KEEP - they say the stored fact is actually still true / has not really changed\n"
        "DELETE - they say it is no longer true and give no replacement value\n"
        "UPDATE: <new fact> - they confirm it has changed; write the NEW fact as one short "
        "sentence in the third person beginning with 'The user', based on what they said above\n"
        "UNRELATED - the reply is not an answer about that fact at all",
        max_tok=60,
    )
    head = (out or "").strip()
    up = head.upper()
    if up.startswith("UPDATE"):
        new = head.split(":", 1)[1].strip() if ":" in head else ""
        # Change confirmed but no clean new value parsed: do NOT fall back to KEEP — that would
        # re-stamp (and thus entrench) a memory 3c already proved contradicted. HOLD = leave it.
        return ("UPDATE", new) if new else ("HOLD", None)
    if up.startswith("DELETE"):
        return "DELETE", None
    if up.startswith("KEEP"):
        return "KEEP", None
    return "UNRELATED", None   # explicit UNRELATED or anything unparseable -> don't trap


def resolve_clarify(user_input, backup_fn=None):
    """Resolve a parked clarify with the user's next-turn answer. Returns True if the turn was
    consumed (an answer was acted on + acknowledged), False if the caller should fall through
    (the input was a command, or not an answer). Mirrors main._pending_recall: never traps the
    user. backup_fn (main._backup_memory_store) runs before any mutation so it's reversible."""
    global _pending_clarify
    pend = _pending_clarify
    if pend is None:
        return False
    # Let slash-commands / exit/clear words through untouched: don't spend an LLM call and keep
    # the clarify parked so the user can answer it after the command.
    stripped = (user_input or "").strip()
    if stripped.startswith("/") or stripped.lower().rstrip(" .!?") in (
        "exit", "quit", "schließen", "schliessen", "clear", "reset", "löschen", "vergessen",
    ):
        return False
    action, new_text = _classify_clarify_answer(
        user_input, pend["memory_text"], pend.get("user_claim", ""))
    if action == "UNRELATED":
        _pending_clarify = None
        print("[CLARIFY] reply not an answer — pending cleared, continuing normally.", flush=True)
        return False
    if action == "HOLD":
        # Change implied, but no clean replacement value could be written. Leave the memory exactly
        # as-is (no re-stamp, no delete) rather than entrench a known-contradicted fact.
        _pending_clarify = None
        print("[CLARIFY] HOLD — change implied but no clean new value parsed; memory untouched.",
              flush=True)
        return False
    old = pend["memory_text"]
    if backup_fn is not None:
        try:
            backup_fn()
        except Exception as e:  # noqa: BLE001
            print(f"[CLARIFY] store backup failed ({type(e).__name__}: {e}); aborting resolve.",
                  flush=True)
            return False
    if action == "KEEP":
        ok = _memory.reconfirm(old)
        print(f"[CLARIFY] KEEP — re-stamped confirmed_at (found={ok}): {old[:60]}", flush=True)
    elif action == "DELETE":
        ok = _memory.delete_by_text(old)
        print(f"[CLARIFY] DELETE (found={ok}): {old[:60]}", flush=True)
    else:  # UPDATE
        ok = _memory.replace(old, new_text)
        print(f"[CLARIFY] UPDATE (found={ok}): '{old[:40]}' -> '{new_text[:40]}'", flush=True)
    _pending_clarify = None
    _respond_clarify_ack(user_input, action, old, new_text, ok)
    return True


def _humanize_fact(text):
    """Strip the stored 3rd-person 'The user ' prefix + trailing period so a memory fact reads
    cleanly inside a spoken/printed confirmation. Pure string op — keeps the receipt faithful."""
    t = (text or "").strip()
    for p in ("The user ", "the user "):
        if t.startswith(p):
            t = t[len(p):]
            break
    return t.rstrip(". ")


def _clarify_receipt(action, old_text, new_text, found):
    """The DETERMINISTIC confirmation of what the store mutation actually did — built purely from
    (action, found), never from the model, so it can NEVER claim something the store didn't do
    (the whole point of splitting confirmation from ack). Returns (display, spoken): display
    carries a ✓/✗ marker for the console; spoken is glyph-free for TTS. A found=False op (target
    not in the store) reports 'nothing changed' instead of a false success."""
    if not found:
        verb = {"UPDATE": "update", "DELETE": "remove", "KEEP": "re-confirm"}.get(action, "change")
        return (f"✗ Nothing changed — I couldn't find that note to {verb}.",
                f"Actually, nothing changed there — I couldn't find that note to {verb}.")
    if action == "UPDATE":
        fact = _humanize_fact(new_text)
        return (f"✓ Updated — your note now reads: {fact}.",
                f"Done — your note now reads: {fact}.")
    if action == "DELETE":
        fact = _humanize_fact(old_text)
        return (f"✓ Removed: {fact}.",
                f"Done — I've let go of the note that you {fact}.")
    # KEEP
    fact = _humanize_fact(old_text)
    return (f"✓ Kept (still true): {fact}.",
            "Done — I've kept that note as it was, and re-confirmed it's still true.")


def _respond_clarify_ack(user_input, action, old_text, new_text, found):
    """Acknowledge a resolved clarify in TWO separated layers (see the four-principle memory
    doctrine — 'verify transparently'): a fact-free persona WARMTH line generated by the model,
    then a DETERMINISTIC receipt built from the actual mutation result. The model never states
    what changed, so its known drift (live 2026-06-08: it 'confirmed' an update that hadn't
    happened) can no longer misreport the store. NO auto-save, NO new clarify."""
    global chat_history, _exchange_count
    runtime_state.set_flag("thinking", True)
    display_receipt, spoken_receipt = _clarify_receipt(action, old_text, new_text, found)
    # 1) Persona warmth ONLY — explicitly fact-free; every specific is carried by the receipt.
    cue = ("[Memory resolved] You have just finished taking care of a memory note at the user's "
           "request. In ONE short, warm sentence, let them know you've handled it. Do NOT state "
           "what changed, do NOT mention any specific fact, do NOT list memories — an exact "
           "confirmation is shown to them separately right after you speak.")
    sys_prompt = _build_system("", _context_summary, cue=cue)
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_input},
    ]
    warm = remove_reasoning(_stream_reply(messages, temperature=0.3)).strip()
    # 2) Surface the deterministic receipt, set apart from the warmth: console (with marker) and
    #    TTS (glyph-free, priority slot) so the FACT the user hears is store-derived, not generated.
    print(f"  {display_receipt}", flush=True)
    tts_emit(spoken_receipt, source="clarification", display=False)
    full_text = (warm + "\n" + display_receipt).strip() if warm else display_receipt
    chat_history.append({"role": "user",      "content": user_input})
    chat_history.append({"role": "assistant", "content": full_text})
    _exchange_count += 1
    _compact_history()
    _save_history()


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

def _stream_reply(messages, temperature=TEMPERATURE, return_suggestion=False):
    """Stream a chat completion to the console + incremental per-sentence TTS, and
    return the RAW full text (caller applies remove_reasoning). Shared by
    respond_streaming and respond_recall so the token/TTS loop lives in one place.
    Assumes the 'thinking' flag is already set by the caller; clears it once the
    handshake returns and manages the 'speaking' flag itself.

    [new-3]: a trailing `SUGGEST_SEARCH: <query>` line is suppressed from the stream (never
    spoken/displayed) and stripped from the returned text. With return_suggestion=True the
    return is (clean_text, suggested_query_or_None) for the live path; otherwise just the text
    (warmup / recall callers, which never trigger the marker)."""
    with _stage("create(stream) handshake"):
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=temperature,
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
        # [new-3]: once the SUGGEST_SEARCH marker line begins, stop feeding console + TTS so the
        # machine-readable suggestion never leaks; it's parsed off the returned text below.
        if _suppress_suggest(full_text):
            continue
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
    answer, suggested = _extract_suggestion(full_text)
    if return_suggestion:
        return answer, suggested
    return answer


def respond_streaming(user_input, llm_input=None, tool_driven=False, grounding_evidence=None,
                      allow_research_arm=True):
    global chat_history, _exchange_count

    # llm_input: callers (e.g. web search) send augmented content to the model while
    # only the original user_input is stored in history — prevents web-result bloat.
    # tool_driven: deterministic tool answers (web search) are barred from long-term memory.
    # grounding_evidence: the source text a web answer is built over; when set, the generated
    # answer is grounding-checked against it (log-only, see _web_answer_grounding).
    llm_text = llm_input if llm_input is not None else user_input

    runtime_state.set_flag("thinking", True)

    # Input-side ack FIRST: speak it now so it plays during the work below.
    with _stage("fire_ack"):
        _fire_ack(user_input)

    with _stage("memory.query"):
        mem_hits = _memory.query(user_input)
    with _stage("pre_generate"):
        pre = stabilizer.pre_generate(user_input, mem_hits)
    clarify_cue   = _maybe_arm_clarify(user_input, mem_hits, pre)
    role_cue = _apply_role_directive()
    with _stage("apply_emotion"):
        emo_cue = _apply_emotion(user_input, pre)
    with _stage("apply_confidence"):
        conf_cue = _apply_confidence(user_input, pre)
    with _stage("memory.context_block"):
        context_block = _memory.context_block(user_input)
    ack_cue       = _ack_cue()
    cue           = " ".join(p for p in (pre.system_cue, clarify_cue, role_cue, emo_cue, conf_cue, ack_cue) if p)
    sys_prompt    = _build_system(context_block, _context_summary, cue=cue)

    messages = [
        {"role": "system", "content": sys_prompt},
        *chat_history[-(MAX_EXCHANGES * 2):],
        {"role": "user",   "content": llm_text}
    ]

    full_text, suggested_query = _stream_reply(messages, return_suggestion=True)

    full_text = remove_reasoning(full_text)
    if not full_text:
        full_text = t('no_response', user_input)

    # [new-3] B2: the model proposed a better search (answer wasn't in the page). Arm a re-search
    # offer UNLESS this turn is itself a re-search (loop cap) or a contradiction-clarify is already
    # pending this turn (don't stack two questions / collide pending states).
    if suggested_query and allow_research_arm and pending_clarify() is None:
        _arm_research(suggested_query)
    elif suggested_query:
        print(f"[RESEARCH] suggestion NOT armed (allow={allow_research_arm}, "
              f"clarify_pending={pending_clarify() is not None}): {suggested_query[:60]!r}",
              flush=True)

    post = stabilizer.post_generate(user_input, full_text, pre, tool_driven=tool_driven)
    _close_emotion(post.clean_text)

    if grounding_evidence:
        _web_answer_grounding(grounding_evidence, post.clean_text)

    chat_history.append({"role": "user",      "content": user_input})
    chat_history.append({"role": "assistant", "content": post.clean_text})
    _exchange_count += 1
    _compact_history()          # doctrine: compress overflow into summary, THEN trim
    _save_history()

    if post.commit_to_memory:
        _auto_save_memory(user_input, post.clean_text)

    return None


def respond_recall(user_input, mode):
    """Answer a 'what do you remember'-style query scoped to ONE channel, so the
    boundary the user asked about is honoured (see RECALL_MODES_PLAN §4b):
      RECALL_LTM      → the long-term store ONLY (full snapshot, no retrieval filter)
      RECALL_CONTEXT  → the rolling summary + recent history ONLY (no stored profile)
      RECALL_COMBINED → all three, explicitly framed
    A recall turn NEVER auto-saves (it is the over-store path; the model reciting
    memory must not seed new entries — see [[vivianna_autosave_judge_gate]])."""
    global chat_history, _exchange_count

    runtime_state.set_flag("thinking", True)

    if mode == RECALL_LTM:
        entries = _memory.all_entries()
        if entries:
            lines = ["[Long-term memories — the ONLY facts you have saved about the user:]"]
            lines += [f"- {e['text']}" for e in entries]
            context_block = "\n".join(lines)
        else:
            context_block = "[Long-term memories: nothing is saved yet.]"
        summary = ""
        history = []   # LTM mode: do not let recent turns bleed in
        instruction = ("[Recall — long-term memory] The user is asking what you have saved "
                       "in long-term memory. List ONLY the saved facts shown above. Add nothing, "
                       "infer nothing, and do NOT mention the current or recent conversation. "
                       "If nothing is saved, say so plainly.")
    elif mode == RECALL_CONTEXT:
        context_block = ""   # no stored profile in a "what were we talking about" answer
        summary = _context_summary
        history = chat_history[-(MAX_EXCHANGES * 2):]
        instruction = ("[Recall — this conversation] The user is asking what you two have been "
                       "talking about in this conversation. Describe only what was discussed here "
                       "(the summary and recent turns). Do NOT state saved profile facts about the user.")
    else:  # RECALL_COMBINED
        context_block = _memory.context_block(user_input)
        summary = _context_summary
        history = chat_history[-(MAX_EXCHANGES * 2):]
        instruction = ("[Recall — everything] The user wants both what you have saved long-term "
                       "and what you two were just discussing. Distinguish the two: say what is "
                       "stored about them versus what came up in this conversation.")

    sys_prompt = _build_system(context_block, summary, cue=instruction)
    messages = [
        {"role": "system", "content": sys_prompt},
        *history,
        {"role": "user", "content": user_input},
    ]

    # Low temperature: a recall answer should be faithful, not creative.
    full_text = _stream_reply(messages, temperature=0.2)
    full_text = remove_reasoning(full_text)
    if not full_text:
        full_text = t('no_response', user_input)

    chat_history.append({"role": "user",      "content": user_input})
    chat_history.append({"role": "assistant", "content": full_text})
    _exchange_count += 1
    _compact_history()
    _save_history()
    # NB: deliberately NO _auto_save_memory here — recall must not seed memory.
    return None


def respond(user_input, llm_input=None, tool_driven=False, grounding_evidence=None):
    global chat_history, _exchange_count

    # llm_input: see respond_streaming — keeps web-augmented text out of history.
    # tool_driven: see respond_streaming — bars deterministic tool answers from memory.
    # grounding_evidence: see respond_streaming — log-only web-answer grounding check.
    llm_text = llm_input if llm_input is not None else user_input

    _fire_ack(user_input)   # input-side ack: speak now, before generation
    mem_hits      = _memory.query(user_input)
    pre           = stabilizer.pre_generate(user_input, mem_hits)
    clarify_cue   = _maybe_arm_clarify(user_input, mem_hits, pre)
    role_cue      = _apply_role_directive()
    emo_cue       = _apply_emotion(user_input, pre)
    conf_cue      = _apply_confidence(user_input, pre)
    context_block = _memory.context_block(user_input)
    ack_cue       = _ack_cue()
    cue           = " ".join(p for p in (pre.system_cue, clarify_cue, role_cue, emo_cue, conf_cue, ack_cue) if p)
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

    if grounding_evidence:
        _web_answer_grounding(grounding_evidence, post.clean_text)

    chat_history.append({"role": "user",      "content": user_input})
    chat_history.append({"role": "assistant", "content": post.clean_text})
    _exchange_count += 1
    _compact_history()          # doctrine: compress overflow into summary, THEN trim
    _save_history()

    if post.commit_to_memory:
        _auto_save_memory(user_input, post.clean_text)

    return post.clean_text
