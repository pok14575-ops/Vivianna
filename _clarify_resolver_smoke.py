"""Offline smoke for the clarify RESOLVER fix (2026-06-08).

Bug: "Yes please do so" (a bare confirmation) was classified KEEP, re-stamping a
contradicted memory, because _classify_clarify_answer only saw the stored fact + the
reply — never the triggering claim that holds the new value. Fix threads user_claim in.

Run with the llama.cpp server UP (uses brain._llm_bare). Greedy (top_k=1) so stable.
"""
import brain

# (label, old_stored_fact, user_claim_that_triggered_clarify, next_turn_reply, expected_action)
CASES = [
    ("THE BUG — bare 'yes please do so' must UPDATE, not KEEP",
     "The user prefers winter over summer due to a dislike of the heat.",
     "I like the summer heat and hate winters cold. Especially Autum is where I thrive.",
     "Yes please do so.",
     "UPDATE"),
    ("bare 'Yes' to the same change",
     "The user prefers winter over summer due to a dislike of the heat.",
     "Actually I love the summer heat now and thrive in autumn.",
     "Yes.",
     "UPDATE"),
    ("genuine re-confirmation must KEEP (mood, not a real change)",
     "The user prefers winter over summer due to a dislike of the heat.",
     "Ah, I love a hot summer day like today.",
     "No, I still prefer winter — I was just enjoying today's sun.",
     "KEEP"),
    ("retraction with no replacement must DELETE",
     "The user prefers winter over summer due to a dislike of the heat.",
     "Honestly I don't really have a favourite season anymore.",
     "That's not true about me anymore, just forget it.",
     "DELETE"),
    ("off-topic reply must be UNRELATED (don't trap)",
     "The user prefers winter over summer due to a dislike of the heat.",
     "I like the summer heat now.",
     "Anyway, what's the weather like today?",
     "UNRELATED"),
]

def main():
    print("=" * 78)
    passed = 0
    for label, old, claim, reply, expected in CASES:
        action, new_text = brain._classify_clarify_answer(reply, old, claim)
        ok = (action == expected)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
        print(f"       reply={reply!r}")
        print(f"       expected={expected}  got={action}  new_text={new_text!r}")
        if action == "UPDATE" and new_text:
            low = new_text.lower()
            # The new value must reflect the CLAIM (summer/autumn), not echo the old winter fact.
            sane = ("summer" in low or "autumn" in low) and "prefers winter" not in low
            print(f"       new_text sane (summer/autumn, not winter)? {sane}")
        print("-" * 78)
    print(f"RESULT: {passed}/{len(CASES)} expected-action matches")

if __name__ == "__main__":
    main()
