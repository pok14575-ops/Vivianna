r"""Offline smoke for the two-mode input gate (runtime_state.can_accept_input). Patches
tts_runner.is_enabled / is_idle so no Kokoro model is loaded. Verifies:
  TTS OFF (text mode) — ready iff text output is done (not thinking, not speaking); the audio
    lifecycle (is_idle) is NOT consulted (stays ready even if is_idle() were False).
  TTS ON  (spoken mode) — unchanged: also blocks while audio is playing / flags set.

Run: F:\AI\Vivianna\venv\Scripts\python.exe F:\AI\Vivianna\_input_gate_smoke.py
"""
import runtime_state as rs
import tts_runner

PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


def set_flags(speaking=False, thinking=False, tool_running=False):
    rs.set_flag("speaking", speaking)
    rs.set_flag("thinking", thinking)
    rs.set_flag("tool_running", tool_running)


# ── TTS OFF: gate on text-output-done only ────────────────────────────────────
tts_runner.is_enabled = lambda: False
tts_runner.is_idle    = lambda: False        # would block the old gate; must be IGNORED now

set_flags()
check("TTS off + idle flags clear -> ready (ignores is_idle)", rs.can_accept_input() is True)
set_flags(thinking=True)
check("TTS off + thinking -> not ready", rs.can_accept_input() is False)
set_flags(speaking=True)
check("TTS off + speaking (text still streaming) -> not ready", rs.can_accept_input() is False)
set_flags(tool_running=True)
check("TTS off + a stray tool_running -> still ready (audio-coupled, not text-gating)",
      rs.can_accept_input() is True)

# ── TTS ON: unchanged behaviour ───────────────────────────────────────────────
tts_runner.is_enabled = lambda: True

tts_runner.is_idle = lambda: True
set_flags()
check("TTS on + idle + flags clear -> ready", rs.can_accept_input() is True)

tts_runner.is_idle = lambda: False
set_flags()
check("TTS on + audio still playing (not idle) -> not ready", rs.can_accept_input() is False)

tts_runner.is_idle = lambda: True
set_flags(speaking=True)
check("TTS on + speaking -> not ready", rs.can_accept_input() is False)
set_flags(tool_running=True)
check("TTS on + tool_running -> not ready", rs.can_accept_input() is False)

set_flags()
print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
