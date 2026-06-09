import time
import threading
import tts_runner

_lock = threading.Lock()
_state = {
    "speaking":     False,
    "listening":    False,
    "thinking":     False,
    "tool_running": False,
    "last_emotion": "calm",
}


def set_flag(key: str, value) -> None:
    with _lock:
        _state[key] = value


def get_flag(key: str):
    with _lock:
        return _state[key]


def snapshot() -> dict:
    with _lock:
        return dict(_state)


# ── Item 1: last-web-context cache (follow-up inheritance) ────────────────────
# After a web turn we stash what was fetched so a CHAT-routed follow-up that continues the same
# topic can be grounded over the real article instead of free-associating (live 2026-06-08
# turn 4: "speculations on locations?" routed to chat and invented wedding venues). Kept here,
# not in brain, so it survives the rolling-history window (a Python var, not transcript tokens).
_web_cache = None


def set_web_cache(topic: str, title: str, url: str, content: str, date=None) -> None:
    global _web_cache
    with _lock:
        _web_cache = {
            "topic": topic, "title": title, "url": url, "content": content,
            "date": date, "ts": time.time(),
        }


def get_web_cache():
    """Most recent grounded web context, or None. Returns a copy (caller can't mutate state)."""
    with _lock:
        return dict(_web_cache) if _web_cache else None


def clear_web_cache() -> None:
    global _web_cache
    with _lock:
        _web_cache = None


def can_accept_input() -> bool:
    # Two modes (2026-06-08):
    #   TTS ON  — unchanged: also wait for audio playback + tail to finish, so the mic never
    #             captures Vivianna's own speech.
    #   TTS OFF — text-only: there is NO audio to overlap, so the gate keys purely on TEXT
    #             OUTPUT being done (not mid-generation, not mid-stream). The audio lifecycle
    #             (tts_runner.is_idle) is intentionally NOT consulted here — that coupling is
    #             what locked keyboard input when TTS was off.
    s = snapshot()
    if not tts_runner.is_enabled():
        return not s["thinking"] and not s["speaking"]
    return (
        tts_runner.is_idle()
        and not s["speaking"]
        and not s["thinking"]
        and not s["tool_running"]
    )


def wait_for_input_ready(poll_interval: float = 0.02, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if can_accept_input():
            return
        time.sleep(poll_interval)
