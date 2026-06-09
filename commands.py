import re


# ── recall modes ───────────────────────────────────────────────────────────────
# Mode constants for "what do you remember"-style queries. They live here (not in
# brain.py) because commands.py is the lowest module in the import graph — it pulls
# in only `re`, so brain.py and main.py can import these without a circular import.
# See RECALL_MODES_PLAN_2026-06-06.md §2.
RECALL_LTM       = "RECALL_LTM"        # answer from the long-term store ONLY
RECALL_CONTEXT   = "RECALL_CONTEXT"    # answer from the rolling summary + recent history ONLY
RECALL_COMBINED  = "RECALL_COMBINED"   # answer from all three channels (explicitly framed)
RECALL_AMBIGUOUS = "RECALL_AMBIGUOUS"  # not a mode — signals "needs A/B/C clarify"


def is_exit_command(text):
    return text.lower() in ["schließen", "schliessen", "exit", "quit"]


def is_switch_to_text_command(text):
    """Spoken (or typed) '3108' — a safe-word that drops out of voice mode into
    keyboard input. Needed before /audit, whose y/N confirmations must be typed.
    Matches the bare code only (whole utterance), tolerating ASR spacing/punctuation
    and the common word transcriptions."""
    if not text:
        return False
    t = text.strip().lower().rstrip(".!?")
    cleaned = re.sub(r"[\s.,!?\-]", "", t)
    if cleaned == "3108":
        return True
    return t in ("thirty-one oh eight", "thirty one oh eight",
                 "three one zero eight", "three one oh eight")


def is_clear_command(text):
    # Tolerate a leading slash ("/reset", "/clear") — these were falling through to
    # the LLM and faking a reset without clearing history.
    return text.strip().lstrip("/").lower() in ["clear", "reset", "löschen", "vergessen"]


def parse_read_command(text):
    """Detect a /read trigger ('/read X', 'read me X', 'read X to me', 'read that to me')
    and return {'source', 'query'} or None. No NLI — direct dispatch. Source defaults to
    'tavily'; 'wikipedia'/'wiki' or 'exa' keywords override. An empty/pronoun query is the
    'read the last thing' sentinel (query == '')."""
    t = text.strip()
    low = t.lower()
    if low.startswith("/read"):
        rest = t[5:].strip()
    elif re.match(r"read\b", low):
        rest = t[4:].strip()
    else:
        return None

    rest_low = rest.lower()
    if "wikipedia" in rest_low or "wiki" in rest_low:
        source = "wikipedia"
    elif "exa" in rest_low:
        source = "exa"
    else:
        source = "tavily"

    # Strip source directives ("on/from/via wikipedia", bare "exa") and read-aloud filler.
    q = re.sub(r"\b(on|from|via|using|in)\s+(wikipedia|wiki|exa)\b", " ", rest, flags=re.I)
    q = re.sub(r"\b(wikipedia|wiki|exa)\b", " ", q, flags=re.I)
    q = re.sub(r"^\s*(please\s+)?(to me|for me|me)\b", " ", q, flags=re.I)
    q = re.sub(r"\b(to me|for me|aloud|out loud)\b", " ", q, flags=re.I)
    q = re.sub(r"\s+", " ", q).strip(" ,.:-\t")

    if q.lower() in ("", "that", "this", "it"):
        q = ""   # read-the-last-thing sentinel
    return {"source": source, "query": q}


# ── recall detector ────────────────────────────────────────────────────────────
# Deterministic dispatcher for "what do you remember"-style queries, mirroring
# parse_read_command (NOT an NLI tool — tool_registry.TOOLS only has web/chat, and a
# regex matcher is higher-confidence here than the 9B classifier). Routes to a recall
# mode, or returns None to fall through to normal routing. See RECALL_MODES_PLAN §4a.
#
# Be deliberately conservative: a *content-bearing* recall ("what do you remember about
# the trip we planned", "do you remember when I said X") must NOT be hijacked — it falls
# through (None) so the normal path can answer with full context. We only catch the bare /
# profile / conversation-scoped forms.

# LTM-scoped: the user is asking about the saved profile / stored facts.
_RECALL_LTM_RES = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bwhat (do|did|does) you (remember|know) about (me|myself|us)\b",
        r"\bwhat (have|did) you (saved?|stored?|remember(ed)?|memoriz(e|ed))\b",
        r"\bwhat(?:’s|'s| is| do you have)?\b.{0,25}\bin your (long[- ]?term )?memory\b",
        r"\bwhat (do|did) you have (saved|stored|on me|about me|in memory|on file)\b",
        r"\bwhat(?:’s|'s| is) (saved|stored)\b.{0,20}\b(about|on) (me|us)\b",
        r"\bwhat do you have on (me|us)\b",
    )
]

# Context-scoped: the user is asking about the current conversation / just now.
_RECALL_CONTEXT_RES = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bwhat (were|are|was) we (just )?(talking about|discussing|doing|saying|on about|chatting about)\b",
        r"\bwhat (did|have) we (just )?(been )?(talk(ed|ing)? about|discuss(ed|ing)?|sa(y|id))\b",
        r"\brecap (our|this|the|that) (conversation|chat|discussion|talk)\b",
        r"\bremind me what we (?:were|(?:just )?(?:were|been))\b",
        r"\bwhat (was|were) we (just )?(on|up to)\b",
    )
]

# Bare recall with no qualifier — genuinely ambiguous, needs A/B/C clarify. Anchored to
# the END of the (de-punctuated) utterance: a bare recall ENDS with "...what do you
# remember/recall" and has no object after the verb. Anchoring the end (not the front)
# lets arbitrary natural preamble through ("All right, now I need you to tell me what do
# you remember") while still rejecting content-bearing recalls, which end in their object
# ("...remember when I said X" ends in "X", not "remember"). A short whitelist of trailing
# TEMPORAL adverbs is permitted ("...remember now / so far / yet") — they qualify *when*,
# not *what*, so the query stays bare (the live 2026-06-08 miss: "...remember now" fell
# through to chat, which then narrated the profile from history instead of the store).
_RECALL_BARE_RE = re.compile(
    r"\bwhat (?:do |did )?you (remember|recall)( anything)?"
    r"(?: (?:now|already|yet|currently|so far|by now|at this point|right now|at the moment|about all this))?"
    r"$",
    re.IGNORECASE,
)


def classify_recall(text):
    """Classify a recall query into a mode, or return None if it isn't one.
    Returns RECALL_LTM / RECALL_CONTEXT / RECALL_AMBIGUOUS, or None (fall through)."""
    if not text:
        return None
    bare = text.strip().rstrip(" .!?…")
    # LTM and CONTEXT are the specific, qualified forms — check them before the bare form.
    for rx in _RECALL_LTM_RES:
        if rx.search(text):
            return RECALL_LTM
    for rx in _RECALL_CONTEXT_RES:
        if rx.search(text):
            return RECALL_CONTEXT
    if _RECALL_BARE_RE.search(bare):
        return RECALL_AMBIGUOUS
    return None


# ── recall clarify answer ──────────────────────────────────────────────────────
# Maps the user's reply to the A/B/C clarify question onto a concrete recall mode.
# Returns None when it is NOT a recognisable choice — the caller must then NOT trap the
# user (it treats the input as a fresh turn). See RECALL_MODES_PLAN §4c step 2.
# A = long-term memory, B = this conversation, C = both.
_CHOICE_LTM_RE  = re.compile(
    r"\b(option a|long[- ]?term|in your memory|from memory|memory|saved|stored|"
    r"about me|my profile|profile|first(?: one)?|number one)\b", re.IGNORECASE)
_CHOICE_CTX_RE  = re.compile(
    r"\b(option b|just talking|talking about|conversation|discuss(?:ed|ion)?|"
    r"this chat|the chat|just now|recent\w*|earlier|second(?: one)?|number two)\b", re.IGNORECASE)
_CHOICE_COMB_RE = re.compile(
    r"\b(option c|both|everything|all of (?:it|them)|combined|third(?: one)?|number three)\b",
    re.IGNORECASE)


def parse_recall_choice(text):
    """Resolve an A/B/C clarify reply to RECALL_LTM / RECALL_CONTEXT / RECALL_COMBINED,
    or None if it isn't a choice. Collect-then-choose: if the reply signals both A and B
    (or says 'both'/'c'), that's COMBINED — a single boolean would mis-resolve it."""
    if not text:
        return None
    low = text.strip().lower().rstrip(" .!?…")
    letters = re.sub(r"[^a-z]", "", low)
    # Whole-utterance single-letter / phonetic answers ("a", "bee", "see").
    if letters in ("a", "ay"):
        return RECALL_LTM
    if letters in ("b", "be", "bee"):
        return RECALL_CONTEXT
    if letters in ("c", "see", "cee"):
        return RECALL_COMBINED
    ltm  = bool(_CHOICE_LTM_RE.search(low))
    ctx  = bool(_CHOICE_CTX_RE.search(low))
    comb = bool(_CHOICE_COMB_RE.search(low))
    if comb or (ltm and ctx):
        return RECALL_COMBINED
    if ltm:
        return RECALL_LTM
    if ctx:
        return RECALL_CONTEXT
    return None


# Explicit-source requests in natural language ("Zusammenfassung von X auf Wikipedia",
# "summarize X from wikipedia", "was sagt Wikipedia über X"). These are NOT caught by
# parse_read_command (which needs a literal "read"/"/read") and the NLI router tends to
# mislabel them as plain chat — so they get hallucinated instead of fetched. This detects
# the source + topic so the router can fetch and ground the answer. Distinct from /read:
# the answer is generated over fetched text (honours "in 6 points" etc.), not read verbatim.

# "exa"/"tavily"/"wiki" must be whole words — bare substrings hide in "Texas", "example".
_SOURCE_RE = re.compile(r"\b(wikipedia|wiki|exa|tavily)\b", re.IGNORECASE)

# Fetch/lookup intent verbs (en + de). Used to gate against casual source mentions.
_FETCH_INTENT_RE = re.compile(
    r"\b(zusammenfass\w*|fasse?|summ?ar(?:y|ise|ize)|was sagt|what does|tell me about|"
    r"erkl[äa]r\w*|erz[äa]hl\w*|gib mir|lies|nachschlagen|schau\w*\s+nach)\b",
    re.IGNORECASE,
)

# Prepositions that, immediately before the source word, signal "… on/from <source>".
_SOURCE_PREP_RE = re.compile(r"\b(auf|von|on|from|via|über|ueber|about|in)$", re.IGNORECASE)


def _extract_topic(text):
    """Strip intent verbs, the source phrase, and trailing formatting directives so the
    fetch query is just the topic. The model still receives the ORIGINAL message, so
    formatting asks ('in 6 Punkten') are preserved — this only feeds the lookup."""
    q = text
    q = _FETCH_INTENT_RE.sub(" ", q)
    # Drop trailing formatting directives ("in 6 Punkten zu …", "in 6 points …").
    q = re.sub(r"\bin\s+\d+\s+(punkten?|stichpunkten?|points?|bullets?)\b.*$", " ", q, flags=re.I)
    # Remove the source phrase with any leading preposition and TLD ("auf Wikipedia.de").
    q = re.sub(r"\b(auf|von|on|from|via|in|über|ueber|about)?\s*(wikipedia|wiki|exa|tavily)(\.\w+)?\b",
               " ", q, flags=re.I)
    # Strip leftover leading prepositions/articles.
    q = re.sub(r"^\s*(von|vom|of|about|über|ueber|on|the|der|die|das)\b", " ", q, flags=re.I)
    q = re.sub(r"[\s,.:;|()\-]+", " ", q).strip()
    return q


def parse_source_request(text):
    """Detect an explicit-source fetch request and return {'source','query'} or None.
    Fires only when a source word is named AND there's fetch intent (a lookup verb, or a
    preposition right before the source) — so casual 'I read it on Wikipedia' is ignored."""
    m = _SOURCE_RE.search(text)
    if not m:
        return None
    kw = m.group(1).lower()
    source = "exa" if kw == "exa" else "tavily" if kw == "tavily" else "wikipedia"

    prep_before = bool(_SOURCE_PREP_RE.search(text[:m.start()].rstrip()))
    if not (prep_before or _FETCH_INTENT_RE.search(text)):
        return None

    query = _extract_topic(text)
    if not query:
        return None
    req = {"source": source, "query": query}

    # An explicit Wikipedia TLD ("wikipedia.de") is an unambiguous language directive.
    # Honour it over detect_language(), which is unreliable on short/German text and was
    # sending "Erdbeere" to en.wikipedia (→ wrong article). Generic TLDs (org/com/net)
    # carry no language signal, so we leave those to the detector.
    if source == "wikipedia":
        tld = re.search(r"\bwikipedia\.([a-z]{2,3})\b", text, re.IGNORECASE)
        if tld and tld.group(1).lower() not in ("org", "com", "net"):
            req["lang"] = tld.group(1).lower()
    return req