"""Offline test for commands.classify_recall — pure regex, no llama-server needed.
Run: python test_classify_recall.py
"""
from commands import (
    classify_recall, parse_recall_choice,
    RECALL_LTM, RECALL_CONTEXT, RECALL_COMBINED, RECALL_AMBIGUOUS,
)

# (input, expected) — None means "fall through to normal routing"
CASES = [
    # --- RECALL_LTM: the saved profile / stored facts ---
    ("what do you remember about me",            RECALL_LTM),
    ("What do you know about me?",               RECALL_LTM),
    ("what have you saved",                       RECALL_LTM),
    ("what have you stored about me",            RECALL_LTM),
    ("what's in your long-term memory",          RECALL_LTM),
    ("what is in your memory",                    RECALL_LTM),
    ("what do you have saved",                    RECALL_LTM),
    ("what do you have on me",                    RECALL_LTM),
    ("what's saved about me",                     RECALL_LTM),
    ("what do you remember about us",            RECALL_LTM),

    # --- RECALL_CONTEXT: the current conversation / just now ---
    ("what were we just talking about",          RECALL_CONTEXT),
    ("what were we talking about",                RECALL_CONTEXT),
    ("what did we just discuss",                  RECALL_CONTEXT),
    ("what were we doing",                        RECALL_CONTEXT),
    ("recap our conversation",                    RECALL_CONTEXT),
    ("recap this chat",                           RECALL_CONTEXT),
    ("remind me what we were talking about",     RECALL_CONTEXT),
    ("what were we up to",                        RECALL_CONTEXT),

    # --- RECALL_AMBIGUOUS: bare, needs clarify ---
    ("what do you remember",                      RECALL_AMBIGUOUS),
    ("What do you remember?",                     RECALL_AMBIGUOUS),
    ("what do you recall",                        RECALL_AMBIGUOUS),
    ("so what do you remember",                   RECALL_AMBIGUOUS),
    ("hey what do you remember",                  RECALL_AMBIGUOUS),
    ("can you tell me what you remember",        RECALL_AMBIGUOUS),
    # preamble-heavy bare recall — the live 2026-06-06 miss (fell through to chat,
    # bled in 'Sa-na-ling-ba' from the summary). End-anchored regex now catches it.
    ("All right, now I need you to tell me what do you remember?", RECALL_AMBIGUOUS),
    ("ok so just tell me what do you recall",    RECALL_AMBIGUOUS),
    ("Vivianna, what do you remember",           RECALL_AMBIGUOUS),
    # trailing TEMPORAL adverb — the live 2026-06-08 miss: fell through to chat, which
    # narrated the autumn profile from history while the store still said winter, hiding
    # the resolver bug. Must classify so the A→LTM path reads the store verbatim.
    ("Tell me what you remember now.",           RECALL_AMBIGUOUS),
    ("what do you remember now",                  RECALL_AMBIGUOUS),
    ("what do you recall so far",                 RECALL_AMBIGUOUS),
    ("what do you remember at this point",       RECALL_AMBIGUOUS),
    ("so what do you remember yet",               RECALL_AMBIGUOUS),

    # --- None: must NOT be hijacked (content-bearing or unrelated) ---
    ("do you remember when I said I love sushi", None),
    ("remember that I have a dentist appointment", None),
    ("what do you remember about the trip we planned", None),
    ("do you remember the time we talked about France", None),
    ("what's the weather today",                  None),
    ("tell me a joke",                            None),
    ("what time is it",                           None),
    ("",                                          None),
]


# A/B/C clarify-answer resolution. None = "not a choice" (must not trap the user).
CHOICE_CASES = [
    ("A",                                RECALL_LTM),
    ("a",                                RECALL_LTM),
    ("long-term",                        RECALL_LTM),
    ("the long term memory one",         RECALL_LTM),
    ("what you saved about me",          RECALL_LTM),
    ("the first one",                    RECALL_LTM),
    ("B",                                RECALL_CONTEXT),
    ("just what we were talking about",  RECALL_CONTEXT),
    ("the conversation",                 RECALL_CONTEXT),
    ("what we discussed just now",       RECALL_CONTEXT),
    ("the second one",                   RECALL_CONTEXT),
    ("C",                                RECALL_COMBINED),
    ("both",                             RECALL_COMBINED),
    ("everything",                       RECALL_COMBINED),
    ("both the memory and the chat",     RECALL_COMBINED),  # A+B signals -> COMBINED
    ("the third one",                    RECALL_COMBINED),
    # not a choice -> None (fall through as fresh turn, don't trap)
    ("actually never mind",              None),
    ("what's the weather",               None),
    ("yeah",                             None),
    ("",                                 None),
]


def main():
    passed = failed = 0
    for text, expected in CASES:
        got = classify_recall(text)
        ok = got == expected
        passed += ok
        failed += not ok
        if not ok:
            print(f"[FAIL classify] {text!r}\n        expected={expected} got={got}")
    for text, expected in CHOICE_CASES:
        got = parse_recall_choice(text)
        ok = got == expected
        passed += ok
        failed += not ok
        if not ok:
            print(f"[FAIL choice] {text!r}\n        expected={expected} got={got}")
    print(f"\n{passed}/{passed + failed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if main() else 1)
