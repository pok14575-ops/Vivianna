import re
import time
from tools_lang import (
    get_task_acknowledgement, get_clarification, get_web_apology, detect_language,
    get_identity_refusal,
)
from web_search import (
    get_web_context, fetch_wikipedia, fetch_tavily, fetch_exa, score_relatedness,
)
from config import (
    WEB_FOLLOWUP_ENABLED, WEB_FOLLOWUP_MAX_AGE_S, WEB_FOLLOWUP_RELATEDNESS_FLOOR,
    WEB_FOLLOWUP_DEBUG, WEB_SOURCE_DATE_ENABLED,
)
from brain import (
    nli_classify, save_memory, purge_memory, set_route_confidence, role_check,
    last_user_message, pending_research, clear_pending_research,
)
from output_bus import emit
from tool_registry import TOOLS
from commands import parse_source_request
import runtime_state

_LANG_INSTRUCTION = {
    "de": "Antworte auf Deutsch.",
    "zh": "请用中文回答。",
    "en": "Answer in English.",
}

# [new-3] search self-correction: when the fetched page does NOT contain the answer, the model
# proposes a better query on a final machine-readable line. brain._stream_reply suppresses that
# line from speech/console and parses the query out to offer a re-search. Keep it terse so the
# 9B reliably places it last and never narrates it.
_SUGGEST_INSTRUCTION = (
    " If the information here does NOT contain the answer, say so plainly but do NOT ask the user "
    "whether to search again (I offer that myself). Then add one final line exactly as: "
    "SUGGEST_SEARCH: <a concrete web search query, plain keywords, NO quotes, likely to find it>. "
    "Do not read out, mention, or explain that line."
)


def _source_note(context):
    """Item 4: tell the model the source identity + publish date so it can (a) flag staleness and
    (b) answer provenance questions honestly instead of inventing. Empty when there's no url."""
    url = (context.get("url") or "").strip()
    if not url:
        return ""
    date = context.get("date")
    if date:
        return (f" This came from {url}, published {date}; if that date is old for a "
                f"time-sensitive question, say the information may be out of date.")
    return f" This came from {url} (no publish date detected; do not claim it is recent)."

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


# ── Item 2: referential / meta query repair ──────────────────────────────────
# A follow-up like "look it up on a reliable source" or "tell me more" carries no topic of its
# own — the real subject lived in the PRIOR turn. Searching the literal text fetches the wrong
# page (live 2026-06-08 turn 6: "look it up on a reliable source" → fetched the Wikipedia
# article "Reliable sources" instead of strawberry nutrition; only the honesty layer saved it).
# We detect these and rebuild the query from the previous user turn. CONSERVATIVE by design:
# over-rewriting a standalone query is the failure mode, so a query carrying its own concrete
# topic ("look up the capital of Brazil") passes through unchanged. English-only for now; a
# non-match degrades safely to the original behaviour. Rule-based (cheap); an LLM reformulation
# is the noted future upgrade. The reformulation half is reused by items 1 and [new-3].

# Whole-query referential commands: a verb + a bare pronoun object (it/that/this), optionally
# trailed by a meta-source/politeness clause ("on a reliable source", "please", "now"). The
# pronoun object is the discriminator — "look IT up" matches, "look up the population" does not.
_REFERENTIAL_RE = re.compile(
    r"^\s*(?:please\s+|could\s+you\s+|can\s+you\s+|would\s+you\s+|pls\s+|hey\s+|so\s+|"
    r"and\s+|now\s+|just\s+|ok(?:ay)?\s+)*"
    r"(?:"
    r"look\s+(?:it|that|this|those|them)\s+up"
    r"|look\s+up\s+(?:it|that|this)"
    r"|search\s+(?:for\s+)?(?:it|that|this|them|more)"
    r"|google\s+(?:it|that|this)"
    r"|find\s+(?:out\s+)?(?:more|it|that|this)"
    r"|check\s+(?:it|that|this)"
    r"|verify\s+(?:it|that|this)"
    r"|confirm\s+(?:it|that|this)"
    r"|(?:tell|give|show)\s+me\s+more"
    r"|(?:what|how)\s+about\s+(?:it|that|this)"
    r"|more\s+(?:on|about)\s+(?:it|that|this)"
    r"|(?:it|that|this)"            # bare pronoun as the whole query
    r")"
    # optional trailing meta-source / politeness clauses, repeatable
    r"(?:[\s,]+(?:on|from|using|with|via|in|please|now|then|instead)"
    r"(?:\s+(?:a|an|the))?"
    r"(?:\s+(?:reliable|trustworthy|credible|reputable|proper|good|better|real|actual|"
    r"official|different|other))*"
    r"(?:\s+(?:sources?|websites?|sites?|places?|internet|web|net))?)*"
    r"[\s.!?]*$",
    re.IGNORECASE,
)

# Shared building blocks for the source-redirect patterns below. Broadened after live run 3
# (2026-06-08) missed every real phrasing: "Maybe try a more reliable source?" (adjective
# "more" before "reliable"), "Yes take more reputable news please" ("take" + "news"), "Please
# look up on reuters" (no pronoun, names a source).
_LEAD = (r"(?:please\s+|could\s+you\s+|can\s+you\s+|would\s+you\s+|maybe\s+|just\s+|"
         r"yes\s+|yeah\s+|ok(?:ay)?\s+|sure\s+|and\s+|so\s+|now\s+|pls\s+|hey\s+)*")
_LOOKUP_VERB = (r"(?:look\s+(?:it|that|this|them\s+)?up|look|search(?:\s+for)?|google|find|"
                r"check|verify|confirm|try|take|get|use|consult|pull\s+up|pull|grab)")
_QUALITY = (r"(?:reliable|trustworthy|credible|reputable|trusted|proper|good|better|real|"
            r"actual|official|legit|legitimate|serious|different|other|another|alternative)")
_SRC_NOUN = r"(?:sources?|news|outlets?|websites?|sites?|articles?|pages?|places?|media|press)"
# A short curated list of commonly-named sources. Brittle by nature; extend as needed. Only used
# to recognise "look up on <source>" — the rebuilt query is still the PRIOR topic (honoring the
# named source for an actual fetch is [new-3]/source-fetch territory, not item 2).
_NAMED_SOURCE = (r"(?:reuters|ap\s*news|associated\s+press|the\s+ap|bbc|cnn|npr|pbs|"
                 r"the\s+guardian|guardian|new\s+york\s+times|nyt|the\s+times|washington\s+post|"
                 r"wsj|wall\s+street\s+journal|bloomberg|axios|politico|al\s+jazeera|abc|cbs|"
                 r"nbc|fox|forbes|wired|techcrunch|wikipedia|yahoo|google\s+news)")

# Meta-source redirect: "(try) a (more) reliable source / reputable news / different outlet",
# with or without a leading lookup verb. Required tail = a source NOUN; the constrained middle
# (only specific optional tokens, never ".*") keeps standalone queries from matching.
_META_SOURCE_RE = re.compile(
    r"^\s*" + _LEAD +
    r"(?:" + _LOOKUP_VERB + r"\s+)?"
    r"(?:(?:it|that|this|them|again|instead|elsewhere)\s+)?"
    r"(?:(?:on|from|with|via|in|using|at)\s+)?"
    r"(?:(?:a|an|the|some|any|another)\s+)?"
    r"(?:(?:more|much|even|far|way|a\s+bit)\s+)?"
    r"(?:" + _QUALITY + r"\s+)?"
    r"(?:" + _SRC_NOUN + r")"
    r"(?:[\s,]+(?:please|now|then|instead|again|thanks?))*[\s.!?]*$",
    re.IGNORECASE,
)

# Named-source redirect: "look up on reuters", "search the guardian", "try bbc".
_NAMED_SOURCE_RE = re.compile(
    r"^\s*" + _LEAD + _LOOKUP_VERB +
    r"(?:\s+(?:it|that|this|them))?"
    r"(?:\s+(?:on|from|at|in|via|using|with))?"
    r"\s+(?:the\s+)?" + _NAMED_SOURCE +
    r"(?:[\s,]+(?:please|now|then|instead|again|thanks?))*[\s.!?]*$",
    re.IGNORECASE,
)


def _looks_referential(text):
    t = text.strip()
    return bool(_REFERENTIAL_RE.match(t) or _META_SOURCE_RE.match(t)
                or _NAMED_SOURCE_RE.match(t))


def _has_topic_content(text):
    """A usable search topic needs ≥2 real word tokens (letters, len ≥2) — rejects greetings /
    bare acknowledgements that would make a useless inherited query."""
    return len(re.findall(r"[^\W\d_]{2,}", text, re.UNICODE)) >= 2


def _resolve_search_query(user_input, prior_user_msg):
    """If user_input is a referential/meta follow-up with no topic of its own, return the prior
    user turn's text as the rebuilt search query; otherwise None (search the original
    unchanged). Conservative: standalone queries, an empty/greeting prior, or a prior that is
    itself referential all return None."""
    if not _looks_referential(user_input):
        return None
    topic = (prior_user_msg or "").strip()
    if not topic or _looks_referential(topic) or not _has_topic_content(topic):
        return None
    return topic


# ── Item 4b: provenance follow-up answered from cached source metadata ────────
# Live run 4 (2026-06-08): after a successful web answer, "from what date is your source?" /
# "what's your source?" routed to CHAT and she INVENTED dates, rumour timelines, and outlets she
# never accessed. These provenance questions are answered here from the cached source metadata
# (url / title / publish date), grounded, so she states the real source or honestly admits a
# missing detail instead of confabulating. (A source *redirect* like "try a better source" is a
# re-search, handled by item 2 / [new-3], not here.)
_PROVENANCE_RE = re.compile(
    r"(?:"
    r"what(?:'s| is| was)?\s+(?:your|the)\s+sources?"
    r"|where\s+(?:did|do|does)\s+(?:you|that|this|it)\s+(?:get|got|find|come|read|see)"
    r"|from\s+wh(?:at|ich)\s+(?:date|source|site|website|article|page)"
    r"|wh(?:at|ich)\s+(?:date|source|site|website|outlet|article|page)\b"
    r"|when\s+was\s+(?:that|this|it|the\s+\w+)\s+(?:published|posted|written|reported|from)"
    r"|how\s+(?:recent|old|reliable|trustworthy|credible|current)\s+(?:is|are|was)\b"
    r"|is\s+(?:that|this|your|the)\s+sources?\s+(?:reliable|credible|trustworthy|recent|current)"
    r"|date\s+of\s+(?:the|your|that|this)\s+source"
    r"|how\s+do\s+you\s+know\b"
    r")",
    re.IGNORECASE,
)


def _is_provenance_question(text):
    return bool(_PROVENANCE_RE.search(text or ""))


def _maybe_provenance(user_input):
    """The cached web context to answer a provenance question from, or None. Same freshness window
    as the follow-up cache. Gated on WEB_SOURCE_DATE_ENABLED (item 4)."""
    if not WEB_SOURCE_DATE_ENABLED or not _is_provenance_question(user_input):
        return None
    cache = runtime_state.get_web_cache()
    if not cache:
        return None
    if time.time() - cache.get("ts", 0) > WEB_FOLLOWUP_MAX_AGE_S:
        return None
    return cache


def _do_provenance(user_input, respond, cache):
    """Answer a 'what/when/how-reliable is your source' question from the cached metadata only —
    state the real URL + publish date (or honestly that there's none), and forbid inventing
    dates / timelines / other outlets. tool_driven keeps it out of long-term memory."""
    url   = (cache.get("url") or "").strip() or "an unknown source"
    title = (cache.get("title") or "").strip()
    date  = cache.get("date")
    lang      = detect_language(user_input)
    lang_note = _LANG_INSTRUCTION.get(lang, "Answer in English.")
    facts = (f"URL: {url}\nTitle: {title}\nPublish date: {date or 'unknown / not detected'}")
    date_str = f"published {date}" if date else "with no detectable publish date"
    augmented = (
        f"{user_input}\n\n"
        f"[Answer the user's question about your source using ONLY these facts. Say plainly that "
        f"the last answer came from {url}, {date_str}. Do NOT invent a date, a timeline of how a "
        f"rumour spread, or any other outlet/source — if a detail isn't in these facts, say you "
        f"don't have it. {lang_note}]\n"
        f"{facts}"
    )
    print(f"[ROUTER] Provenance question -> answering from cached source {url!r} (date={date})",
          flush=True)
    return respond(user_input, llm_input=augmented, tool_driven=True, grounding_evidence=facts)


# ── [new-3] B2: search self-correction — confirm & execute ───────────────────
_AFFIRM_RE = re.compile(
    r"^\s*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|please|pls|alright|right|fine|absolutely|"
    r"definitely|of\s+course|go\s+ahead|go\s+for\s+it|do\s+it|try\s+it|sounds\s+good|"
    r"yes\s+please|please\s+do)\b",
    re.IGNORECASE,
)
_NEGATE_RE = re.compile(
    r"^\s*(?:no|nope|nah|don'?t|do\s+not|stop|cancel|never\s*mind|forget\s+it|leave\s+it)\b",
    re.IGNORECASE,
)


def _is_affirmative(text):
    """Cheap deterministic yes-detector for a re-search offer. Conservative: a clear 'no' or any
    non-affirmative reply returns False so the offer is dropped and the turn falls through to
    normal routing (never traps the user). Mirrors the clarify 'never trap' rule."""
    t = (text or "").strip()
    if not t or _NEGATE_RE.match(t):
        return False
    return bool(_AFFIRM_RE.match(t))


def resolve_research(user_input, respond):
    """Resolve a parked [new-3] re-search offer with the user's next-turn reply. Returns True if
    the turn was consumed (the suggested query was executed), False to fall through. A confirmed
    re-search runs through the web path with allow_research_arm=False (loop cap). Slash-commands /
    exit-words pass through with the offer still parked, like resolve_clarify."""
    pend = pending_research()
    if pend is None:
        return False
    stripped = (user_input or "").strip()
    if stripped.startswith("/") or stripped.lower().rstrip(" .!?") in (
        "exit", "quit", "schließen", "schliessen", "clear", "reset", "löschen", "vergessen",
    ):
        return False
    if not _is_affirmative(user_input):
        clear_pending_research()
        print("[RESEARCH] reply not a yes — offer dropped, continuing normally.", flush=True)
        return False
    query = pend["query"]
    clear_pending_research()
    print(f"[RESEARCH] confirmed -> executing re-search: {query[:70]!r}", flush=True)
    _do_web(query, respond, allow_research_arm=False)   # loop cap: this turn cannot re-arm
    return True


# ── Item 1: web follow-up inheritance ─────────────────────────────────────────
def _maybe_web_followup(user_input):
    """Return the cached web context to re-ground over when this CHAT-routed turn continues the
    last web topic, else None. Two triggers: a deterministic REFERENTIAL cue (acts now), or a
    cross-encoder RELATEDNESS score above the (calibrate-live, default-inert) floor for content
    follow-ups that carry no referential marker. Relatedness is always logged. Fallback-safe."""
    if not WEB_FOLLOWUP_ENABLED:
        return None
    cache = runtime_state.get_web_cache()
    if not cache:
        return None
    if time.time() - cache.get("ts", 0) > WEB_FOLLOWUP_MAX_AGE_S:
        runtime_state.clear_web_cache()        # stale → drop, behave as a fresh chat turn
        return None
    referential = _looks_referential(user_input)
    rel = score_relatedness(user_input, cache["topic"])
    gate = rel is not None and rel >= WEB_FOLLOWUP_RELATEDNESS_FLOOR
    if WEB_FOLLOWUP_DEBUG:
        rel_s = f"{rel:.3f}" if rel is not None else "n/a"
        print(f"[WEB-FOLLOWUP] relatedness={rel_s} floor={WEB_FOLLOWUP_RELATEDNESS_FLOOR} "
              f"referential={referential} act={referential or gate} "
              f"topic={cache['topic'][:50]!r}", flush=True)
    return cache if (referential or gate) else None


def _do_followup_inherit(user_input, respond, cache):
    """Answer a continuing follow-up by GROUNDING over the already-fetched article (no new
    network call). The honesty instruction ('say you couldn't find it rather than guessing or
    speculating') is what turns the un-answerable case into honest deferral instead of the
    invented-venues confabulation — and a wrong inherit degrades to the same honest 'not here'."""
    lang      = detect_language(user_input)
    lang_note = _LANG_INSTRUCTION.get(lang, "Answer in English.")
    augmented = (
        f"{user_input}\n\n"
        f"[Continuing from the web article below — answer using ONLY this information. If it "
        f"does not actually contain the answer, say you could not find it rather than guessing "
        f"or speculating.{_SUGGEST_INSTRUCTION} {lang_note}]\n"
        f"Title: {cache['title']}\n\n"
        f"{cache['content']}"
    )
    print(f"[ROUTER] Web follow-up -> re-grounding over cached article {cache['url']!r}",
          flush=True)
    return respond(user_input, llm_input=augmented, tool_driven=True,
                   grounding_evidence=cache['content'])


def _do_web(user_input, respond, search_query=None, allow_research_arm=True):
    ack = get_task_acknowledgement("web", user_input)
    emit(ack, source="acknowledgement")

    # search_query (item 2) lets a referential follow-up search the inherited topic while the
    # user-facing prompt/answer still reflect what the user actually said.
    # allow_research_arm ([new-3] loop cap) is False when THIS call is itself a confirmed
    # re-search, so a second miss can't arm another offer.
    topic = search_query or user_input
    context = get_web_context(topic)
    if context:
        # Item 1: cache the grounded context so a continuing CHAT follow-up can re-ground over
        # this article instead of confabulating. Item 4: cache the publish date too.
        runtime_state.set_web_cache(topic, context['title'], context['url'],
                                    context['content'], date=context.get('date'))
        lang      = detect_language(user_input)
        lang_note = _LANG_INSTRUCTION.get(lang, "Answer in English.")
        augmented = (
            f"{user_input}\n\n"
            f"[Web search result — answer the question using ONLY the information below. "
            f"If it does not actually contain the answer, say you could not find it rather "
            f"than guessing.{_source_note(context)}{_SUGGEST_INSTRUCTION} {lang_note}]\n"
            f"Title: {context['title']}\n\n"
            f"{context['content']}"
        )
        return respond(user_input, llm_input=augmented, tool_driven=True,
                       grounding_evidence=context['content'],
                       allow_research_arm=allow_research_arm)

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
            f"Do not add facts that are not present here.{_SUGGEST_INSTRUCTION} {lang_note}]\n"
            f"Title: {result.get('title', '')}\n\n"
            f"{result['content']}"
        )
        return respond(user_input, llm_input=augmented, tool_driven=True,
                       grounding_evidence=result['content'])

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
        resolved = _resolve_search_query(user_input, last_user_message())
        if resolved:
            print(f"[ROUTER] Referential query repaired: {user_input!r} -> {resolved!r}",
                  flush=True)
        return _do_web(user_input, respond, search_query=resolved)

    # 2a. Item 4b: a provenance question about the last web answer ("what's your source / from
    # what date") → answer from cached source metadata, not a confabulated date/timeline.
    prov = _maybe_provenance(user_input)
    if prov:
        set_route_confidence(confidence)
        return _do_provenance(user_input, respond, prov)

    # 2b. Item 1: a continuing web follow-up that NLI did NOT route to a fresh search → re-ground
    # over the cached article rather than free-associating (the turn-4 invented-venues bug).
    cache = _maybe_web_followup(user_input)
    if cache:
        set_route_confidence(confidence)
        return _do_followup_inherit(user_input, respond, cache)

    if confidence < _NLI_LOW:
        clarification = get_clarification(user_input)
        print(f"[ROUTER] Low confidence ({confidence:.2f}) → clarification", flush=True)
        emit(clarification, source="clarification")
        return None

    # 3. Chat fallback
    print(f"[ROUTER] Chat (tool={tool_name} confidence={confidence:.2f})", flush=True)
    set_route_confidence(confidence)
    return respond(user_input)
