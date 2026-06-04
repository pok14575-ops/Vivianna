from tts import speak, speak_priority

# Sources that route through speak_priority (priority=0 queue slot)
_PRIORITY_SOURCES = frozenset({"acknowledgement", "clarification", "apology", "system"})


def emit(text: str, source: str = "llm", speak_out: bool = True, display: bool = True) -> None:
    if not text or not text.strip():
        return
    if display:
        print(f"\nVivianna: {text}", flush=True)
    if speak_out:
        if source in _PRIORITY_SOURCES:
            speak_priority(text)
        else:
            speak(text)
