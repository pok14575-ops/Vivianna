from tts_runner import enqueue, wait_idle, is_enabled
from tools_lang import detect_language

# Per-language Kokoro voice. English synthesises via espeak G2P (lang code passed to
# Kokoro); Chinese synthesises via misaki phonemes (handled in tts_runner) — lang="zh"
# is the routing marker the synth worker watches for.
EN_VOICE = "af_heart"
ZH_VOICE = "zf_xiaobei"


def _route(text: str) -> tuple[str, str]:
    """Pick (voice, lang) from the sentence's own language. Per-sentence, so a reply
    can switch EN<->ZH mid-stream and each chunk gets the right voice."""
    if detect_language(text) == "zh":
        return ZH_VOICE, "zh"
    return EN_VOICE, "en-us"


def speak(text: str, voice: str | None = None, lang: str | None = None):
    if not is_enabled():
        return
    if text and text.strip():
        if voice is None or lang is None:
            voice, lang = _route(text)
        enqueue(text, voice, lang, priority=10)


def speak_priority(text: str, voice: str | None = None, lang: str | None = None):
    if not is_enabled():
        return
    if text and text.strip():
        if voice is None or lang is None:
            voice, lang = _route(text)
        print("[TTS] priority queued", flush=True)
        enqueue(text, voice, lang, priority=0)
