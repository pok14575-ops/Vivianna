import re
from tools_lang import (
    get_task_acknowledgement, get_clarification, get_web_apology, detect_language,
    get_identity_refusal,
)
from web_search import get_web_context, fetch_wikipedia, fetch_tavily, fetch_exa
from brain import nli_classify, save_memory, purge_memory, set_route_confidence, role_check
from output_bus import emit
from tool_registry import TOOLS
from commands import parse_source_request

_LANG_INSTRUCTION = {
    "de": "Antworte auf Deutsch.",
    "zh": "请用中文回答。",
    "en": "Answer in English.",
}

# NLI confidence thresholds
_WEB_HIGH = 0.90
_NLI_LOW  = 0.70

_REMEMBER_RE = re.compile(
    r"^(?:please\s+)?(?:"
    r"remember\s*[:]?\s*(?:that\s+|this[,:]?\s+)?"
    r"|merke\s+dir[,:]?\s*(?:dass\s+)?"
    r"|bitte\s+merken[,:]?\s*(?:dass\s+)?"
    r"|记住[：:]?\s*"
    r")(.*)",
    re.IGNORECASE | re.DOTALL,
)

_PURGE_PHRASES = {
    "purge memory", "purge memories", "purge all memories",
    "gedächtnis löschen", "speicher löschen", "alles vergessen",
    "清除记忆", "删除记忆",
}

# Current time/date questions. Real local time is injected into every system prompt,
# so these must be answered by normal generation — NEVER a web search, which wastes a
# call and lets a scraped page fight the injected time (observed: 9B picked the wrong
# year). Patterns are deliberately narrow ("now" intent only); a miss just falls through
# to NLI routing. No \b — it does not behave around CJK characters.
_TIME_QUERY_RE = re.compile(
    r"(?:what(?:'s| is)? the time|what time is it|current time|local time|time right now"
    r"|what(?:'s| is)? the date|today'?s date|current date|what day is it"
    r"|wie sp(?:ä|ae)t|uhrzeit|ortszeit|welches datum|welcher tag ist|datum heute"
    r"|几点|现在时间|今天几号|今天星期几)",
    re.IGNORECASE,
)


def _do_web(user_input, respond):
    ack = get_task_acknowledgement("web", user_input)
    emit(ack, source="acknowledgement")

    context = get_web_context(user_input)
    if context:
        lang      = detect_language(user_input)
        lang_note = _LANG_INSTRUCTION.get(lang, "Answer in English.")
        augmented = (
            f"{user_input}\n\n"
            f"[Web search result — use this to answer naturally. {lang_note}]\n"
            f"Title: {context['title']}\n\n"
            f"{context['content']}"
        )
        return respond(user_input, llm_input=augmented, tool_driven=True)

    apology = get_web_apology(user_input)
    print(f"[DDG] No results.", flush=True)
    emit(apology, source="apology")
    return respond(user_input)


_WIKI_LANGS = {"en", "de", "zh"}


def _do_source_fetch(req, user_input, respond):
    """Explicit-source request (e.g. 'Zusammenfassung von X auf Wikipedia'): fetch the
    named source, then GENERATE the answer over that text (so 'in 6 points' etc. work).
    tool_driven=True keeps these deterministic lookups out of long-term memory."""
    source, query = req["source"], req["query"]
    ack = get_task_acknowledgement("web", user_input)
    emit(ack, source="acknowledgement")

    if source == "wikipedia":
        # Explicit TLD (req["lang"], e.g. "wikipedia.de") wins. Otherwise detect from the
        # FULL message (not the single-word topic), clamped to a Wikipedia we support.
        lang = req.get("lang")
        if not lang:
            detected = detect_language(user_input)
            lang = detected if detected in _WIKI_LANGS else "en"
        result = fetch_wikipedia(query, lang)
    elif source == "exa":
        result = fetch_exa(query)
    else:
        result = fetch_tavily(query)

    if result and result.get("content", "").strip():
        lang      = detect_language(user_input)
        lang_note = _LANG_INSTRUCTION.get(lang, "Answer in English.")
        augmented = (
            f"{user_input}\n\n"
            f"[{source.capitalize()} source — answer the request using ONLY the text below. "
            f"Do not add facts that are not present here. {lang_note}]\n"
            f"Title: {result.get('title', '')}\n\n"
            f"{result['content']}"
        )
        return respond(user_input, llm_input=augmented, tool_driven=True)

    apology = get_web_apology(user_input)
    print(f"[FETCH] No content from {source} for {query!r}.", flush=True)
    emit(apology, source="apology")
    return respond(user_input)


def route(user_input, respond):
    stripped = user_input.strip()
    lc       = stripped.lower()

    # 0a. Purge memory
    if lc in _PURGE_PHRASES or lc.startswith("purge mem"):
        purge_memory()
        lang = detect_language(user_input)
        msg  = "Memory purged." if lang == "en" else "Gedächtnis geleert." if lang == "de" else "记忆已清除。"
        emit(msg, source="system")
        return None

    # 0b. Remember command
    m = _REMEMBER_RE.match(stripped)
    if m:
        fact = m.group(1).strip().rstrip(".")
        if fact:
            save_memory(fact)
            lang = detect_language(user_input)
            msg  = "Got it, I'll remember that." if lang == "en" else "Gemerkt." if lang == "de" else "已记住。"
            emit(msg, source="system")
        return None

    # 1. Role / identity preservation — may refuse, or flag 'cautious' for generation
    # (Time/date is no longer a tool: real local time is injected into every system
    #  prompt, so Vivianna answers it naturally during normal generation.)
    decision = role_check(user_input)
    if decision == "refuse":
        print("[ROUTER] Role refusal — request would violate intended role", flush=True)
        emit(get_identity_refusal(user_input), source="system")
        return None

    # 1b. Current time/date → answer via injected time during normal generation, never
    # web search. Deterministic guard, ahead of NLI (see _TIME_QUERY_RE).
    if _TIME_QUERY_RE.search(stripped):
        print("[ROUTER] Time/date query → injected-time chat (no web)", flush=True)
        set_route_confidence(1.0)
        return respond(user_input)

    # 1c. Explicit-source request ("… auf Wikipedia", "summarize … from wikipedia"): the
    # NLI router mislabels these as chat and the model hallucinates a "source". Force a
    # real fetch + grounded generation instead. Deterministic guard, ahead of NLI.
    source_req = parse_source_request(stripped)
    if source_req:
        print(f"[ROUTER] Source request → fetch {source_req['source']} "
              f"q={source_req['query']!r}", flush=True)
        set_route_confidence(1.0)
        return _do_source_fetch(source_req, user_input, respond)

    # 2. NLI classification
    tool_name, confidence = nli_classify(user_input)

    if tool_name == "web" and confidence >= _WEB_HIGH:
        print(f"[ROUTER] Web search (confidence={confidence:.2f})", flush=True)
        set_route_confidence(confidence)
        return _do_web(user_input, respond)

    if confidence < _NLI_LOW:
        clarification = get_clarification(user_input)
        print(f"[ROUTER] Low confidence ({confidence:.2f}) → clarification", flush=True)
        emit(clarification, source="clarification")
        return None

    # 3. Chat fallback
    print(f"[ROUTER] Chat (tool={tool_name} confidence={confidence:.2f})", flush=True)
    set_route_confidence(confidence)
    return respond(user_input)
