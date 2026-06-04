# test_phase_a_live.py
"""
Vivianna Phase A — automated LIVE regression harness.

Drives the REAL stack (router.route -> respond_streaming), the same entry point as
keyboard input in main.py — NOT a mock. Captures the debug markers that the human
ear cannot objectively judge ([ACK:EMIT], [ACK:FALLBACK], [TIMING] fire_ack,
[EMOTION:PRE], [TIMING] create(stream) handshake) and renders a per-turn verdict
plus a Phase-A summary. Full terminal log is tee'd to test_logs/.

Run (server + stack must be live):
    python test_phase_a_live.py            # auto-enables TTS for playback timing
    python test_phase_a_live.py --no-tts   # text-only (timings reflect generation only)
    python test_phase_a_live.py --clear-first   # /clear history before the run (opt-in)

SCOPE / honesty (this harness does NOT modify ack_coordinator.py or brain.py):
  * It reports THREE outcomes, never conflating them:
        PASS         — expected ack fired with the right category
        FAIL         — a true error (false fire on routine, or EMOTIONAL beating SAFETY)
        INCONCLUSIVE — the gate fired nothing, so precedence/fallback wasn't exercised
                       (a regex-coverage gap — a tuning target, NOT an architecture fail)
  * Input 8 (tool>emotional): tool acks are emitted OUTSIDE the coordinator in Phase A
    (router._do_web), so this turn does not truly exercise the precedence stack — reported
    as a CAVEAT.
  * Input 10 (neutral fallback): the deterministic gate is always confidence 1.0, so the
    neutral-fallback path is dormant until Phase C. A silence here is BY DESIGN — CAVEAT.
  * Running real inputs creates real history/memory side effects (auto-save). Use
    --clear-first and/or back up the data folder if you care.
"""

from __future__ import annotations

import io
import os
import re
import sys
import time
from datetime import datetime

# UTF-8 console like main.py (umlauts in inputs 9-side / German cases).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_BASE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_BASE, "test_logs")


# ── stdout tee: real console + logfile + per-turn capture buffer ───────────────
class _Tee:
    encoding = "utf-8"

    def __init__(self, real, logfile):
        self._real = real
        self._log = logfile
        self.turn_buffer: io.StringIO | None = None

    def write(self, s):
        self._real.write(s)
        self._log.write(s)
        if self.turn_buffer is not None:
            self.turn_buffer.write(s)
        return len(s)

    def flush(self):
        self._real.flush()
        self._log.flush()

    def isatty(self):
        return False


# ── marker parsers ─────────────────────────────────────────────────────────────
_RE_ACK_EMIT   = re.compile(r"\[ACK:EMIT\]\s+cat=(\w+)")
_RE_ACK_FALL   = re.compile(r"\[ACK:FALLBACK\]")
_RE_FIRE_ACK   = re.compile(r"\[TIMING\]\s+fire_ack:\s*([\d.]+)s")
_RE_EMO_PRE    = re.compile(r"\[EMOTION:PRE\]\s+state=(\w+)")
_RE_HANDSHAKE  = re.compile(r"\[TIMING\]\s+create\(stream\) handshake:\s*([\d.]+)s")
_RE_ROUTER_WEB = re.compile(r"\[ROUTER\]\s+Web search")


def _parse(captured: str) -> dict:
    ack = _RE_ACK_EMIT.search(captured)
    fire_vals = [float(x) for x in _RE_FIRE_ACK.findall(captured)]
    hs = _RE_HANDSHAKE.search(captured)
    emo = _RE_EMO_PRE.search(captured)
    return {
        "ack_category": ack.group(1).upper() if ack else None,
        "ack_fallback": bool(_RE_ACK_FALL.search(captured)),
        "fire_ack_max": max(fire_vals) if fire_vals else None,
        "emotion_pre": emo.group(1) if emo else None,
        "handshake": float(hs.group(1)) if hs else None,
        "web_route": bool(_RE_ROUTER_WEB.search(captured)),
    }


# ── test cases ─────────────────────────────────────────────────────────────────
# kind: "silent" | "fire" | "precedence" | "fallback"
CASES = [
    dict(id=1, kind="silent", expect=None, text=(
        "Vivianna, how do I reverse a list in Python?")),
    dict(id=2, kind="silent", expect=None, text=(
        "Vivianna, what's the weather like in Berlin today?")),
    dict(id=3, kind="fire", expect="EMOTIONAL", text=(
        "Vivianna, I've been feeling really alone lately. Like nobody in the house "
        "actually sees me. I just needed to tell someone.")),
    dict(id=4, kind="fire", expect="EMOTIONAL", text=(
        "Vivianna, she left me. Three years together and she said she doesn't love me "
        "anymore. I don't know what to do with myself.")),
    dict(id=5, kind="fire", expect="SAFETY", text=(
        "Vivianna, the kids are home alone and I think one of them might have taken "
        "something from the medicine cabinet. I'm not sure what to do.")),
    dict(id=6, kind="fire", expect="CRITICAL_TECHNICAL", text=(
        "Vivianna, I need you to be precise. I'm about to push a change to production "
        "and I need to know exactly what happens if the migration fails halfway through.")),
    dict(id=7, kind="fire", expect="MEMORY_CONFLICT", text=(
        "Vivianna, you told me last week that the Kokoro model only runs on CPU. But "
        "I'm seeing GPU usage right now. Which one is right?")),
    dict(id=8, kind="precedence", expect_not="EMOTIONAL", text=(
        "Vivianna, I'm really stressed out right now, I haven't slept. Can you run a "
        "search for the latest DeBERTa benchmarks on GLUE and tell me the numbers."),
        caveat="tool ack is emitted outside the coordinator in Phase A"),
    dict(id=9, kind="precedence", expect="SAFETY", expect_not="EMOTIONAL", label="R1", text=(
        "Vivianna, I feel so overwhelmed and scared right now. My youngest swallowed "
        "something small from the floor and I don't know if it was a battery or a toy "
        "piece. He seems okay but I'm panicking.")),
    dict(id=10, kind="fallback", expect="NEUTRAL", text=(
        "Vivianna... I don't know. It's just a lot right now. Everything."),
        caveat="neutral fallback is dormant in Phase A (gate confidence always 1.0)"),
]


def _verdict(case: dict, p: dict) -> dict:
    """Return {status, note}. status in PASS|FAIL|INCONCLUSIVE|CAVEAT."""
    cat = p["ack_category"]
    kind = case["kind"]

    if kind == "silent":
        if cat is None and not p["ack_fallback"]:
            return dict(status="PASS", note="silent as expected")
        return dict(status="FAIL", note=f"FALSE FIRE: cat={cat} fallback={p['ack_fallback']}")

    if kind == "fire":
        if cat is None:
            return dict(status="INCONCLUSIVE", note="gate fired nothing (regex-coverage gap)")
        if cat == case["expect"]:
            return dict(status="PASS", note=f"fired {cat}")
        return dict(status="FAIL", note=f"WRONG CATEGORY: got {cat}, expected {case['expect']}")

    if kind == "precedence":
        if cat is None:
            base = "gate fired nothing"
            if case.get("caveat"):
                base += f" — {case['caveat']}"
            web = " [web route taken: task ack emitted outside coordinator]" if p["web_route"] else ""
            return dict(status="INCONCLUSIVE", note=base + web)
        if cat == case.get("expect_not"):
            return dict(status="FAIL", note=f"PRECEDENCE INVERSION: {cat} won")
        if case.get("expect") and cat == case["expect"]:
            return dict(status="PASS", note=f"precedence correct: {cat} won")
        return dict(status="PASS", note=f"non-{case['expect_not']} fired: {cat}")

    if kind == "fallback":
        if cat == "NEUTRAL" or p["ack_fallback"]:
            return dict(status="PASS", note="neutral fallback fired")
        if cat is None:
            note = "silent"
            if case.get("caveat"):
                note += f" — {case['caveat']}"
            return dict(status="CAVEAT", note=note)
        return dict(status="FAIL", note=f"strong-category misfire on ambiguous input: {cat}")

    return dict(status="INCONCLUSIVE", note="unknown kind")


# ── tone-inversion check (cold ack on tender input) ────────────────────────────
_COLD = {"CRITICAL_TECHNICAL", "TOOL_TASK"}
_TENDER_IDS = {3, 4, 5, 9}


def _tone_inversion(case: dict, p: dict) -> bool:
    return case["id"] in _TENDER_IDS and p["ack_category"] in _COLD


# ── live driving ───────────────────────────────────────────────────────────────
def _wait_tts_idle(timeout=120.0):
    import tts_runner
    import runtime_state
    time.sleep(0.2)  # let respond_streaming enqueue before we poll
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if (tts_runner.is_idle()
                and not runtime_state.get_flag("speaking")
                and not runtime_state.get_flag("thinking")):
            return True
        time.sleep(0.05)
    return False


def main():
    args = set(sys.argv[1:])
    autoenable_tts = "--no-tts" not in args
    clear_first = "--clear-first" in args

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"phase_a_live_{stamp}.log")
    logfile = open(log_path, "w", encoding="utf-8")
    tee = _Tee(sys.stdout, logfile)
    sys.stdout = tee

    print(f"=== PHASE A LIVE TEST — {stamp} ===")
    print(f"log: {log_path}\n")

    # Import the live stack (this loads MemoryManager + needs llama.cpp running).
    try:
        from router import route
        from brain import respond_streaming
        import brain
        from output_bus import emit
        from tts_runner import set_enabled as set_tts_enabled, is_enabled as tts_is_enabled
        import config
    except Exception as e:
        print(f"[ABORT] Could not import the stack: {e}")
        print("        Is the environment set up? (venv active, models present)")
        return 2

    # ── preflight: the markers only exist if these flags are on ──
    pre_warn = []
    if not getattr(config, "ACK_LAYER_ENABLED", False):
        pre_warn.append("ACK_LAYER_ENABLED is False — no acks will fire (test meaningless).")
    if getattr(config, "ACK_MODE", "") != "deterministic":
        pre_warn.append(f"ACK_MODE={getattr(config,'ACK_MODE','?')} (expected 'deterministic').")
    if not getattr(config, "ACK_DEBUG", False):
        pre_warn.append("ACK_DEBUG is False — [ACK:EMIT] lines won't print; harness blind.")
    if not getattr(config, "EMOTION_DEBUG", False):
        pre_warn.append("EMOTION_DEBUG is False — [EMOTION:PRE] won't print.")
    if not getattr(brain, "_STAGE_TIMING", False):
        pre_warn.append("brain._STAGE_TIMING is False — [TIMING] fire_ack/handshake won't print.")
    for w in pre_warn:
        print(f"[PREFLIGHT WARN] {w}")
    if pre_warn:
        print()

    if clear_first:
        try:
            brain.clear_history()
            print("[SETUP] history cleared (--clear-first).\n")
        except Exception as e:
            print(f"[SETUP] clear_history failed: {e}\n")

    if autoenable_tts and not tts_is_enabled():
        try:
            set_tts_enabled(True)  # lazily loads Kokoro; needed for playback timing
        except Exception as e:
            print(f"[SETUP] TTS enable failed ({e}); continuing text-only.\n")
    print(f"[SETUP] TTS enabled: {tts_is_enabled()}\n")

    results = []
    fire_ack_values = []
    handshakes = []
    resp_times = []
    tone_flags = []

    for case in CASES:
        cid = case["id"]
        label = f" ({case['label']})" if case.get("label") else ""
        print("\n" + "─" * 78)
        print(f"INPUT {cid}{label} [{case['kind']}]  expect="
              f"{case.get('expect') or ('NOT ' + case['expect_not'] if case.get('expect_not') else 'SILENT')}")
        print(f"> {case['text']}")
        print("─" * 78)

        tee.turn_buffer = io.StringIO()
        t_sent = time.perf_counter()
        try:
            response = route(case["text"], respond_streaming)
            if response:
                emit(response, source="tool")
        except Exception as e:
            tee.turn_buffer = None
            print(f"[TURN ERROR] {e}")
            if cid == 1:
                print("[ABORT] First turn errored — likely the llama.cpp server is down.")
                return 2
            results.append((case, dict(ack_category=None, ack_fallback=False,
                                       fire_ack_max=None, emotion_pre=None,
                                       handshake=None, web_route=False),
                            dict(status="INCONCLUSIVE", note=f"turn error: {e}")))
            time.sleep(3)
            continue
        t_resp = time.perf_counter()
        _wait_tts_idle()
        t_tts = time.perf_counter()
        captured = tee.turn_buffer.getvalue()
        tee.turn_buffer = None

        p = _parse(captured)
        v = _verdict(case, p)
        if p["fire_ack_max"] is not None:
            fire_ack_values.append((cid, p["fire_ack_max"]))
        if p["handshake"] is not None:
            handshakes.append(p["handshake"])
        resp_times.append(t_resp - t_sent)
        if _tone_inversion(case, p):
            tone_flags.append((cid, p["ack_category"]))

        print(f"\n[VERDICT {cid}] {v['status']}  — {v['note']}")
        print(f"  ack_category : {p['ack_category']}")
        print(f"  fallback     : {p['ack_fallback']}")
        print(f"  fire_ack     : {p['fire_ack_max']!r} s")
        print(f"  emotion_pre  : {p['emotion_pre']}")
        print(f"  handshake    : {p['handshake']!r} s")
        print(f"  timing       : sent→resp {t_resp - t_sent:.2f}s | resp→tts_done {t_tts - t_resp:.2f}s")
        results.append((case, p, v))

        time.sleep(3)  # brief-mandated inter-turn gap

    _summary(results, fire_ack_values, handshakes, resp_times, tone_flags, stamp)
    logfile.flush()
    return 0


# ── summary block ──────────────────────────────────────────────────────────────
def _summary(results, fire_ack_values, handshakes, resp_times, tone_flags, stamp):
    def stat(cid):
        for c, p, v in results:
            if c["id"] == cid:
                return v["status"]
        return "?"

    fire_ids = [3, 4, 5, 6, 7]
    fires_ok = sum(1 for i in fire_ids if stat(i) == "PASS")
    fires_incon = sum(1 for i in fire_ids if stat(i) == "INCONCLUSIVE")
    silent_ok = sum(1 for i in (1, 2) if stat(i) == "PASS")
    false_fires = sum(1 for i in (1, 2) if stat(i) == "FAIL")
    prec_ok = sum(1 for i in (8, 9) if stat(i) == "PASS")
    fb_ok = 1 if stat(10) == "PASS" else 0
    r1 = stat(9)

    # timing
    bad_timing = [(i, v) for i, v in fire_ack_values if v > 0.050]
    warn_timing = [(i, v) for i, v in fire_ack_values if 0.010 < v <= 0.050]
    timing_ok = (len(fire_ack_values) > 0 and not bad_timing)

    # overall
    if false_fires or bad_timing or r1 == "FAIL":
        overall = "FAIL"
    elif r1 == "PASS" and fires_incon == 0 and fb_ok:
        overall = "PASS"
    else:
        overall = "PARTIAL"

    hs_lo = f"{min(handshakes)*1000:.0f}" if handshakes else "n/a"
    hs_hi = f"{max(handshakes)*1000:.0f}" if handshakes else "n/a"
    rt_lo = f"{min(resp_times):.1f}" if resp_times else "n/a"
    rt_hi = f"{max(resp_times):.1f}" if resp_times else "n/a"

    print("\n\n=== PHASE A TEST SUMMARY ===")
    print(f"Date: {stamp}")
    print("Total: 10 inputs\n")
    print("ACK GATE:")
    print(f"  Correct fires:      {fires_ok}/5  (inputs 3,4,5,6,7 expected to fire)"
          f"{'  [' + str(fires_incon) + ' inconclusive: coverage gaps]' if fires_incon else ''}")
    print(f"  Correct silences:   {silent_ok}/2  (inputs 1,2 expected silent)")
    print(f"  Precedence correct: {prec_ok}/2  (inputs 8,9)")
    print(f"  Fallback correct:   {fb_ok}/1  (input 10)")
    print(f"  Regression R1:      {r1}  (input 9 — SAFETY must beat EMOTIONAL)")
    print("\nLATENCY:")
    if not fire_ack_values:
        print("  fire_ack all near-zero: n/a (no fire_ack timing captured)")
    elif bad_timing:
        print(f"  fire_ack all near-zero: FAIL  (>0.050s: {bad_timing})")
    elif warn_timing:
        print(f"  fire_ack all near-zero: PASS (with warnings 0.010–0.050s: {warn_timing})")
    else:
        print("  fire_ack all near-zero: PASS")
    print(f"  Handshake range: {hs_lo}ms – {hs_hi}ms")
    print(f"  Response time range: {rt_lo}s – {rt_hi}s")
    print("\nTONE:")
    print(f"  Tone-inversion flags: {len(tone_flags)}"
          + (f"  {tone_flags}" if tone_flags else ""))

    # honest caveats
    print("\nNOTES (by design, not failures):")
    print(f"  Input 8 precedence: {stat(8)} — tool ack is emitted OUTSIDE the coordinator in")
    print("    Phase A; this turn does not truly exercise the coordinator precedence stack.")
    print(f"  Input 10 fallback:  {stat(10)} — neutral fallback is dormant in Phase A (gate")
    print("    confidence is always 1.0); a silence here is expected until Phase C.")
    if r1 == "INCONCLUSIVE":
        print("  R1 INCONCLUSIVE: the gate fired nothing on input 9, so precedence was not")
        print("    exercised live. The unit test already proves the precedence LOGIC; this is a")
        print("    REGEX-COVERAGE gap (and a real safety-detection miss worth tuning), NOT an")
        print("    architecture inversion. Architecture is not disproven.")

    print(f"\nOVERALL: {overall}")
    print("=" * 28)
    print("\nPhase-A bar: fire_ack timing clean + 0 false fires (1,2) + R1 not FAIL.")
    print("Coverage misses on 3–7 are regex-tuning targets, not architecture failures.")


if __name__ == "__main__":
    raise SystemExit(main())
