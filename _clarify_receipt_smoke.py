"""Pure offline test for the deterministic clarify receipt (2026-06-08 ack-split).
No llama-server needed — _clarify_receipt / _humanize_fact are pure functions.

The contract: the receipt is built from (action, found) ONLY, so it can never claim a
mutation the store didn't make. Critically, a found=False op must NOT say it succeeded.
"""
import brain

OLD = "The user prefers winter over summer due to a dislike of the heat."
NEW = "The user prefers summer over winter and thrives in autumn."

SUCCESS_WORDS = ("updated", "removed:", "now reads", "let go", "kept")


def check(label, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {label}")
    return cond


def main():
    passed = total = 0

    # _humanize_fact strips the 3rd-person prefix + trailing period.
    total += 1; passed += check(
        "humanize strips 'The user ' + period",
        brain._humanize_fact(NEW) == "prefers summer over winter and thrives in autumn")

    cases = [
        # (action, found, display_must_have, display_must_NOT_have)
        ("UPDATE", True,  ["✓", "Updated", "prefers summer over winter"], ["✗"]),
        ("DELETE", True,  ["✓", "Removed", "prefers winter over summer"], ["✗"]),
        ("KEEP",   True,  ["✓", "Kept", "still true"],                    ["✗"]),
        # found=False: must report nothing changed, must NOT assert any success word.
        ("UPDATE", False, ["✗", "Nothing changed"], ["✓", "now reads"]),
        ("DELETE", False, ["✗", "Nothing changed"], ["✓", "Removed:"]),
        ("KEEP",   False, ["✗", "Nothing changed"], ["✓"]),
    ]
    for action, found, must, mustnot in cases:
        disp, spoken = brain._clarify_receipt(action, OLD, NEW, found)
        ok = all(m in disp for m in must) and all(m not in disp for m in mustnot)
        total += 1; passed += check(f"{action} found={found} display: {disp!r}", ok)
        # Spoken line must be glyph-free (TTS) ...
        glyph_free = ("✓" not in spoken) and ("✗" not in spoken)
        total += 1; passed += check(f"{action} found={found} spoken glyph-free", glyph_free)
        # ... and a failed op must never speak a success word either.
        if not found:
            safe = not any(w in spoken.lower() for w in SUCCESS_WORDS)
            total += 1; passed += check(f"{action} found=False spoken claims no success", safe)

    print(f"\nRESULT: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
