import sys
# Console is cp1252 by default (the launcher doesn't set UTF-8). Web content and even
# LLM em-dashes/curly-quotes would raise UnicodeEncodeError on print. errors="replace"
# makes all console output crash-proof; chcp 65001 in run_vivianna.bat renders it correctly.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import time
import msvcrt
from brain import respond_streaming, clear_history, print_debug
from tools_lang import t
from commands import is_exit_command, is_clear_command, parse_read_command
from read_aloud import handle_read
from router import route
from output_bus import emit
from runtime_state import wait_for_input_ready, set_flag
from tts_runner import set_enabled as set_tts_enabled, is_enabled as tts_is_enabled
from config import VOICE_INPUT, ASR_POST_DELAY

# Import the local ASR module unconditionally so /asr works even when starting in
# keyboard mode. The Whisper model still lazy-loads on enable, so the import is cheap.
try:
    import asr
    from asr import listen_and_transcribe
    _voice_ok = True
except Exception as e:
    print(f"[ASR] Local engine unavailable — keyboard only: {e}", flush=True)
    _voice_ok = False

_VOICE_ACTIVE      = _voice_ok and VOICE_INPUT
_FAIL_THRESHOLD    = 3
_consecutive_fails = 0

# Starting in voice mode? Load + warm the engine now so the first utterance isn't slow.
if _VOICE_ACTIVE:
    asr.set_enabled(True)


def read_keyboard_input(prompt="You: ", idle_timeout=2.5):
    print(prompt, end="", flush=True)
    user_input      = ""
    started_typing  = False
    last_input_time = None

    while True:
        if msvcrt.kbhit():
            char = msvcrt.getwch()
            if char == "\r":
                user_input += "\n"
                print()
            elif char == "\b":
                if user_input:
                    user_input = user_input[:-1]
                    print("\b \b", end="", flush=True)
            else:
                user_input += char
                print(char, end="", flush=True)
            started_typing  = True
            last_input_time = time.time()

        if started_typing and last_input_time is not None:
            if time.time() - last_input_time >= idle_timeout:
                print()
                return user_input.strip()

        time.sleep(0.01)


def get_input() -> str:
    global _VOICE_ACTIVE, _consecutive_fails

    wait_for_input_ready()   # block until all 4 conditions clear
    set_flag("listening", True)

    if _VOICE_ACTIVE:
        time.sleep(ASR_POST_DELAY)
        text = listen_and_transcribe()

        if text is None:
            _consecutive_fails += 1
            if _consecutive_fails >= _FAIL_THRESHOLD:
                print(
                    f"[ASR] {_FAIL_THRESHOLD} consecutive device errors — "
                    f"switching to keyboard. Type /voice to re-enable.",
                    flush=True,
                )
                _VOICE_ACTIVE = False
            set_flag("listening", False)
            return ""

        _consecutive_fails = 0
        if text:
            print(f"You: {text}", flush=True)
        set_flag("listening", False)
        return text

    text = read_keyboard_input(idle_timeout=2.5)
    set_flag("listening", False)
    return text


while True:
    user_input = get_input()

    if not user_input:
        continue

    if is_exit_command(user_input):
        if not _VOICE_ACTIVE:
            input("Press Enter to close...")
        break

    if is_clear_command(user_input):
        clear_history()
        emit(t('history_cleared', user_input), source="system")
        continue

    if user_input.lower() == "/debug":
        print_debug()
        continue

    if user_input.lower() == "/voice":
        if not _voice_ok:
            print("[ASR] Voice input unavailable (local engine failed to load / no mic).")
        else:
            _VOICE_ACTIVE      = not _VOICE_ACTIVE
            _consecutive_fails = 0
            if _VOICE_ACTIVE:
                asr.set_enabled(True)   # ensure the engine is loaded when switching to mic
            print(f"[ASR] Voice input {'ON' if _VOICE_ACTIVE else 'OFF'}.")
        continue

    if user_input.lower() == "/asr":
        if not _voice_ok:
            print("[ASR] Local engine unavailable.")
        else:
            asr.set_enabled(not asr.is_enabled())   # first enable lazily loads + warms the model
        continue

    if user_input.lower() == "/tts":
        set_tts_enabled(not tts_is_enabled())   # first enable lazily loads the model
        continue

    read_cmd = parse_read_command(user_input)
    if read_cmd is not None:
        handle_read(read_cmd)
        continue

    start    = time.perf_counter()
    response = route(user_input, respond_streaming)
    end      = time.perf_counter()

    # Fallback: if a route path returns text instead of calling emit(), handle it here
    if response:
        emit(response, source="tool")

    print(f"[TIME] Response time: {end - start:.2f} seconds")
