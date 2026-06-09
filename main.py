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
import os
import shutil
from datetime import datetime
from brain import (respond_streaming, respond_recall, clear_history, print_debug,
                   audit_scan, apply_audit, draft_merge,
                   pending_clarify, resolve_clarify, pending_research)
from tools_lang import t
from commands import (is_exit_command, is_clear_command, parse_read_command,
                      is_switch_to_text_command, classify_recall, parse_recall_choice,
                      RECALL_AMBIGUOUS)
from read_aloud import handle_read
from router import route, resolve_research
from output_bus import emit
from runtime_state import wait_for_input_ready, set_flag
from tts_runner import set_enabled as set_tts_enabled, is_enabled as tts_is_enabled
from config import VOICE_INPUT, ASR_POST_DELAY, MEMORY_VECTORS_PATH, MEMORY_META_PATH, ROLLBACK_DIR

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

# Cross-turn state for the recall A/B/C clarify dialog: holds the ORIGINAL bare query
# while we wait for the user's A/B/C answer (None = not waiting). See handle_recall.
_pending_recall = None

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


def _backup_memory_store():
    """Copy the store files to instant rollback/ before an audit mutates them."""
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    rollback = ROLLBACK_DIR
    os.makedirs(rollback, exist_ok=True)
    for src, tag in ((MEMORY_VECTORS_PATH, "memory_vectors"),
                     (MEMORY_META_PATH, "memory_meta")):
        if os.path.exists(src):
            ext = os.path.splitext(src)[1]
            shutil.copy2(src, os.path.join(rollback, f"{tag}_{ts}{ext}"))
    return ts


def _resolve_conflict(p, entries, merges, deletes):
    """Interactive resolver for a CONFLICT cluster. The auto-merger refuses to touch
    a CONFLICT (a fact in one entry may contradict another, or the cluster mixes
    exact dups with messier supersets), so the user drives it by hand here — INSIDE
    the audit session, so directives never leak out to the chat router.

    Commands (operate on the bracketed indices printed above):
        merge <i> <j> ...   draft+confirm one sentence combining those entries
        delete <i> ...      drop those entries
        done                finish this cluster
    A `claimed` set stops the same index being merged and deleted in one pass; any
    leftover overlap simply surfaces again on the next /audit run."""
    print("   -> CONFLICT — resolve manually:")
    print("        merge <i> <j> ...   combine those entries into one")
    print("        delete <i> ...      drop those entries")
    print("        done                finish this cluster (no change)")
    valid = set(p["indices"])
    claimed = set()
    while True:
        parts = input("   conflict> ").strip().split()
        if not parts:
            break
        action, args = parts[0].lower(), [a.strip(",") for a in parts[1:]]
        if action in ("done", "skip", "n", "no"):
            break
        if action not in ("merge", "delete"):
            print("   ? commands: merge / delete / done")
            continue
        try:
            picks = [int(a) for a in args]
        except ValueError:
            print("   ? numeric indices only, e.g. 'merge 4 5'")
            continue
        need = 2 if action == "merge" else 1
        bad = [i for i in picks if i not in valid]
        used = [i for i in picks if i in claimed]
        if len(picks) < need or bad or used:
            note = f"   ? pick {need}+ from {sorted(valid - claimed)}"
            if bad:  note += f"; unknown: {bad}"
            if used: note += f"; already queued: {used}"
            print(note)
            continue
        if action == "merge":
            draft = draft_merge([entries[i]["text"] for i in picks])
            if not draft:
                print("   merge draft failed (LLM unreachable?) — try again or 'done'.")
                continue
            print(f"   proposed: {draft}")
            ans = input("   [y] accept / type your own wording / [n] cancel: ").strip()
            if ans.lower() in ("n", "no", ""):
                print("   cancelled.")
                continue
            text = draft if ans.lower() in ("y", "yes") else ans
            merges.append((picks, text))
            claimed.update(picks)
            print(f"   queued: merge {picks}.")
        else:  # delete
            deletes.extend(picks)
            claimed.update(picks)
            print(f"   queued: delete {picks}.")


def _delete_by_index(entries, merges, deletes):
    """Full-store delete pass: list EVERY entry with its index and let the user
    purge any of them — clustered or not — so a lone bad memory (which never forms
    a cluster, so the verdict loop never surfaces it) can still be removed. Indices
    are positions in all_entries(), the same space as the cluster indices and the
    merge inputs, so apply_audit_batch drops the union safely. Indices already
    consumed by a queued merge are shown as (queued) and refused."""
    if input("\nDelete any entries by index? [y/N] ").strip().lower() != "y":
        return
    claimed = set(deletes)
    for idxs, _ in merges:
        claimed.update(idxs)
    print(f"\n[AUDIT] full store ({len(entries)}):")
    for i, e in enumerate(entries):
        print(f"   [{i}] {e['text']}{'   (queued)' if i in claimed else ''}")
    print("   delete <i> <j> ...   (or just the numbers)   |   done")
    valid = set(range(len(entries)))
    while True:
        parts = input("   delete> ").strip().split()
        if not parts:
            break
        head = parts[0].lower()
        if head in ("done", "skip", "n", "no"):
            break
        nums = parts[1:] if head == "delete" else parts   # accept "delete 4 6" or "4 6"
        try:
            picks = [int(a.strip(",")) for a in nums]
        except ValueError:
            print("   ? numeric indices, e.g. 'delete 4 6' or '4 6'")
            continue
        bad = [i for i in picks if i not in valid]
        used = [i for i in picks if i in claimed]
        if not picks or bad or used:
            note = f"   ? pick from {sorted(valid - claimed)}"
            if bad:  note += f"; unknown: {bad}"
            if used: note += f"; already queued: {used}"
            print(note)
            continue
        deletes.extend(picks)
        claimed.update(picks)
        print(f"   queued: delete {picks}.")


def handle_audit():
    """/audit — report near-duplicate / conflicting memories, propose merges,
    apply only the ones the user confirms (store backed up first).

    Voice-aware: with TTS ON, reviewing a large store would read every memory aloud
    (a 100-entry store could speak for minutes), so she gives a spoken hint to turn
    TTS off and CANCELS. With TTS off, she acknowledges and excuses the brief wait,
    then proceeds. (The y/N confirmations are typed — say '3108' to switch to keyboard.)"""
    if tts_is_enabled():
        emit("Let's not run the audit with my voice on — I'd read every memory aloud "
             "and it could take minutes, so I'm cancelling this one. Turn my voice off "
             "with /tts first (say 3108 if you need to switch to typing), then run "
             "/audit again.", source="system")
        return
    emit("Okay — auditing my memory for duplicates now. Give me a moment; this can "
         "take ten to fifteen seconds while I review everything.", source="system")
    print("[AUDIT] Scanning memory store (cosine cluster -> LLM review)...")
    try:
        proposals, entries = audit_scan()
    except Exception as e:
        print(f"[AUDIT] Scan failed: {type(e).__name__}: {e}")
        return
    print(f"[AUDIT] {len(entries)} memories; "
          f"{len(proposals)} overlapping cluster(s).")
    if not proposals:
        print("[AUDIT] No overlapping clusters — skipping consolidation.")

    merges, deletes = [], []
    for n, p in enumerate(proposals, 1):
        print(f"\n--- cluster {n}/{len(proposals)}  verdict={p['verdict']} ---")
        for idx, tx in zip(p["indices"], p["texts"]):
            print(f"   [{idx}] {tx}")
        if p["verdict"] == "DUPLICATE" and p["merged"]:
            print(f"   -> propose MERGE into:\n      {p['merged']}")
            if input("   Apply this merge? [y/N] ").strip().lower() == "y":
                merges.append((p["indices"], p["merged"]))
                print("   queued.")
            else:
                print("   skipped.")
        elif p["verdict"] == "CONFLICT":
            _resolve_conflict(p, entries, merges, deletes)
        else:
            print(f"   -> {p['verdict']} — left as-is.")

    _delete_by_index(entries, merges, deletes)

    if not merges and not deletes:
        print("\n[AUDIT] No changes queued; store unchanged.")
        return
    print(f"\n[AUDIT] {len(merges)} merge(s), {len(deletes)} delete(s) queued.")
    if input("Apply all queued changes now? [y/N] ").strip().lower() == "y":
        ts = _backup_memory_store()
        new_count = apply_audit(merges=merges, deletes=deletes)
        print(f"[AUDIT] Applied (store backed up @ {ts}). Now {new_count} memories.")
    else:
        print("[AUDIT] Aborted; store unchanged.")


_RECALL_CLARIFY_MSG = (
    "Do you mean: A, what I have saved in long-term memory; "
    "B, what we were just talking about; or C, both?"
)


def handle_recall(user_input, mode):
    """Dispatch a detected recall query (see RECALL_MODES_PLAN §4c). Unambiguous
    LTM/CONTEXT phrasings auto-route to the mode-scoped generator. A bare/ambiguous
    query instead asks the A/B/C clarify question and parks the original query in
    _pending_recall; the next turn's answer is resolved at the top of the main loop."""
    global _pending_recall
    if mode == RECALL_AMBIGUOUS:
        _pending_recall = user_input
        emit(_RECALL_CLARIFY_MSG, source="system")
        return
    respond_recall(user_input, mode)


while True:
    user_input = get_input()

    if not user_input:
        continue

    # Resolve a pending recall clarify (A/B/C). Must run before every other handler.
    # A recognised choice answers in the chosen mode using the ORIGINAL query; anything
    # else clears the pending state and falls through, so the user is never trapped.
    if _pending_recall is not None:
        choice = parse_recall_choice(user_input)
        if choice is not None:
            orig = _pending_recall
            _pending_recall = None
            handle_recall(orig, choice)
            continue
        _pending_recall = None

    # Resolve a pending contradiction clarify (vivianna_contradiction_clarify_plan). The answer
    # was solicited at the end of the previous reply. True = consumed (resolved + acknowledged);
    # False = a command or not-an-answer -> fall through (resolve_clarify keeps the clarify parked
    # for a command, or clears it for unrelated input, mirroring the recall pattern). The store is
    # backed up before any mutation via _backup_memory_store.
    if pending_clarify() is not None:
        if resolve_clarify(user_input, _backup_memory_store):
            continue

    # Resolve a parked [new-3] re-search offer ("want me to search X?"). A yes executes the
    # suggested query through the web path; anything else drops the offer and falls through.
    if pending_research() is not None:
        if resolve_research(user_input, respond_streaming):
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

    # Spoken '3108' — drop out of voice mode into keyboard input (so /audit's typed
    # y/N confirmations are reachable hands-free). Safe-word style, whole-utterance.
    if is_switch_to_text_command(user_input):
        if _voice_ok and (_VOICE_ACTIVE or asr.is_enabled()):
            _VOICE_ACTIVE = False
            asr.set_enabled(False)
            print("[ASR] Voice command '3108' — ASR off, switched to keyboard input.")
        else:
            print("[ASR] Already in keyboard input.")
        continue

    if user_input.lower() == "/tts":
        set_tts_enabled(not tts_is_enabled())   # first enable lazily loads the model
        continue

    if user_input.lower() == "/audit":
        handle_audit()
        continue

    recall_mode = classify_recall(user_input)
    if recall_mode is not None:
        handle_recall(user_input, recall_mode)
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
