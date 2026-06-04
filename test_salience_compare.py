"""
Salience seat-earning test: regex vs DeBERTa semantic scorer.

Scores REALISTIC auto-save candidates — the third-person one-sentence facts the
LLM extractor actually emits (brain._auto_save_memory) — with both judges, at the
live SALIENCE_STORE_THRESHOLD, and reports store/skip accuracy vs ground truth.

Question: does semantic salience produce better memory quality than pattern
matching on the REAL input distribution (which is mostly third-person)?
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from salience_layer import SalienceLayer, SalienceConfig
from config import (SALIENCE_STORE_THRESHOLD, SALIENCE_MODEL_NAME,
                    SALIENCE_MODEL_DEVICE)

TH = SALIENCE_STORE_THRESHOLD

# (text, should_store, note)  — as the extractor emits them: third person.
CANDIDATES = [
    # --- SHOULD STORE (lasting, salient) ---
    ("The user's name is Jamie.", True, "identity, 3rd-person"),
    ("Jamie is a self-taught solo developer building a local AI assistant called Vivianna.", True, "project/identity, paraphrased"),
    ("Jamie's father does not speak English and primarily speaks Chinese.", True, "family identity"),
    ("Jamie is allergic to penicillin.", True, "health/preference, critical"),
    ("Jamie prefers honest, direct feedback over praise.", True, "lasting preference"),
    ("Jamie has a young daughter who recently started school.", True, "family identity"),
    ("Jamie's long-term goal is to publish Vivianna with credit to the AI collaborators.", True, "project goal"),
    ("Jamie's partner of three years recently left him.", True, "major life event"),
    ("The user works as a freelance graphic designer.", True, "occupation, 3rd-person"),
    # --- SHOULD SKIP (transient / noise / task) ---
    ("The weather in Berlin today is 26 degrees and sunny.", False, "transient weather"),
    ("Jamie asked how to reverse a list in Python.", False, "transient coding task"),
    ("It is currently around 9 PM.", False, "transient time"),
    ("Jamie wants the latest DeBERTa benchmark numbers on GLUE.", False, "transient task"),
    ("Jamie said hello and asked how things are going.", False, "noise/greeting"),
    ("Jamie mentioned he is a bit tired this evening.", False, "transient mood"),
]


def evaluate(layer, tag):
    correct = 0
    rows = []
    for text, gt, note in CANDIDATES:
        sr = layer.score(text)
        decision = sr.score >= TH
        ok = (decision == gt)
        correct += ok
        rows.append((text, sr.score, decision, gt, ok, note, sr.reasons))
    return rows, correct


def main():
    print(f"threshold={TH}  model={SALIENCE_MODEL_NAME}  device={SALIENCE_MODEL_DEVICE}\n")

    regex = SalienceLayer(SalienceConfig(debug=False, use_model=False))
    model = SalienceLayer(SalienceConfig(debug=False, use_model=True,
                                         model_name=SALIENCE_MODEL_NAME,
                                         model_device=SALIENCE_MODEL_DEVICE))

    r_rows, r_correct = evaluate(regex, "regex")
    m_rows, m_correct = evaluate(model, "model")

    print(f"{'expect':>6} {'rgx':>5} {'r?':>3} {'mdl':>5} {'m?':>3}  text")
    print("-" * 100)
    for (text, rscore, rdec, gt, rok, note, _), (_, mscore, mdec, _, mok, _, _) in zip(r_rows, m_rows):
        exp = "STORE" if gt else "skip"
        rmark = "OK" if rok else "XX"
        mmark = "OK" if mok else "XX"
        print(f"{exp:>6} {rscore:>5.2f} {rmark:>3} {mscore:>5.2f} {mmark:>3}  {text[:64]}")

    n = len(CANDIDATES)
    print("-" * 100)
    print(f"REGEX  accuracy: {r_correct}/{n}")
    print(f"MODEL  accuracy: {m_correct}/{n}")

    # where they disagree on the store/skip decision
    print("\nDISAGREEMENTS (regex vs model decision):")
    any_dis = False
    for (text, rscore, rdec, gt, _, note, rreasons), (_, mscore, mdec, _, _, _, mreasons) in zip(r_rows, m_rows):
        if rdec != mdec:
            any_dis = True
            print(f"  [{'STORE' if gt else 'skip'}] regex={rscore:.2f}->{'store' if rdec else 'skip'} | "
                  f"model={mscore:.2f}->{'store' if mdec else 'skip'}  ::  {text[:70]}")
    if not any_dis:
        print("  (none)")


if __name__ == "__main__":
    main()
