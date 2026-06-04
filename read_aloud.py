"""/read command handler: fetch a source, speak/print it in order, bookend with
opening + closing lines.

Ordering note (the one real trap in tts_runner): the synth queue is a PriorityQueue
ordered by (priority, seq). Every priority-0 item is pulled before ANY priority-10
item, so mixing a priority-0 opening/closing with priority-10 body sentences would
play "opening, closing, …body". Therefore EVERY part of a /read — opening, each
body sentence, and closing — is emitted at the SAME (normal) priority. source="read"
is not in output_bus._PRIORITY_SOURCES, so it routes through normal speak() and the
itertools seq counter preserves strict insertion order. Never use a priority source here.
"""

import re

from output_bus import emit
from brain import _split_sentences
from tts_preprocess import clean_for_tts
from web_search import fetch_wikipedia, fetch_tavily, fetch_exa
from tools_lang import detect_language
from config import READ_OPENING, READ_CLOSING

# Residual scaffolding that survives extraction (Exa's API text, leftover table cells):
# markdown table-separator lines, pipes, and footnote stars are noise when read aloud.
_TABLE_SEP = re.compile(r"^\s*\|?[\s:|.-]*-{2,}[\s:|.-]*\|?\s*$", re.M)


def _clean_read_text(text):
    text = _TABLE_SEP.sub("", text)
    text = text.replace("∗", "").replace("|", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# en/de/zh have Wikipedias; anything else falls back to English.
_WIKI_LANGS = {"en", "de", "zh"}


def _wiki_lang(query):
    lang = detect_language(query)
    return lang if lang in _WIKI_LANGS else "en"


def _fetch(source, query):
    if source == "wikipedia":
        return fetch_wikipedia(query, _wiki_lang(query))
    if source == "exa":
        return fetch_exa(query)
    return fetch_tavily(query)   # default


def _read_last_assistant():
    """'read that to me' with no topic → re-read the previous assistant turn."""
    import brain
    for m in reversed(brain.chat_history):
        if m.get("role") == "assistant" and m.get("content", "").strip():
            return {"title": "", "url": "", "content": m["content"]}
    return None


def handle_read(cmd):
    source, query = cmd["source"], cmd["query"]

    if not query:
        result = _read_last_assistant()
        if not result:
            emit("I don't have anything to read yet.", source="read")
            return
    else:
        print(f"[READ] source={source} query={query!r}", flush=True)
        result = _fetch(source, query)

    if not result or not result.get("content", "").strip():
        emit("I couldn't find anything to read.", source="read")
        return

    body = _clean_read_text(result["content"])
    if not body:
        emit("I couldn't find anything to read.", source="read")
        return

    # Opening (prints + speaks).
    emit(READ_OPENING, source="read", display=True)

    # Body: print once for text-mode visibility, then push sentences through TTS
    # only (display=False) so the console isn't spammed with per-sentence prefixes.
    # clean_for_tts mirrors the normal speech path (strips markdown/quotes/URLs).
    if result.get("title"):
        print(f"\n[Reading: {result['title']}]", flush=True)
    print(body + "\n", flush=True)
    for sentence in _split_sentences(body):
        spoken = clean_for_tts(sentence)
        if spoken:
            emit(spoken, source="read", display=False)

    # Closing (FIFO after the body — same priority, so it plays last).
    emit(READ_CLOSING, source="read", display=True)
