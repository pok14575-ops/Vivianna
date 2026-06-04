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


def can_accept_input() -> bool:
    s = snapshot()
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
