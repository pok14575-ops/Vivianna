import re


def is_exit_command(text):
    return text.lower() in ["schließen", "schliessen", "exit", "quit"]


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