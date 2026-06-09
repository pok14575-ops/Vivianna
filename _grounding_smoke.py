"""Smoke test for grounding.py (the third-piece NLI singleton) — standalone, pre-wiring.

Proves, in isolation (no runtime touched):
  1. The singleton loads fp16 on CUDA and the label-order guard passes.
  2. classify() separates the three relations on real grounding-shaped pairs.
  3. The two semantic helpers point the right way:
       - contradiction_prob HIGH on a real conflict, LOW on agreement/unrelated.
       - entailment_prob   HIGH on support, LOW on a conflict.
  4. Idempotent prewarm + cached-instance behavior.

Run: venv_tf5\Scripts\python.exe _grounding_smoke.py
"""
import grounding

# (premise/evidence, hypothesis/claim, expected dominant relation)
CASES = [
    ("The user is allergic to peanuts.", "The user can safely eat peanuts.", "contradiction"),
    ("The user is vegan and avoids all animal products.", "The user loves steak.", "contradiction"),
    ("The user lives in Berlin, Germany.", "The user lives in Germany.", "entailment"),
    ("The user is a freelance fashion designer.", "The user works in fashion.", "entailment"),
    ("The user loves skating outside in the park.", "The user enjoys skating in the park.", "entailment"),
    ("The user has a wife named Vivianna.", "The user enjoys hiking on weekends.", "neutral"),
    ("The user prefers tea over coffee.", "The user owns an RTX 5070.", "neutral"),
]


def main():
    print("=== grounding.py smoke ===")
    m, t = grounding.get_model()
    if m is None:
        print("LOAD FAILED — fallback path engaged (see [GROUND] line above). "
              "That IS the fallback-safe behavior, but the model didn't load.")
        return
    print(f"loaded on {grounding._dev}; label guard passed "
          f"(e/n/c = {grounding.ENTAIL}/{grounding.NEUTRAL}/{grounding.CONTRADICT})\n")

    ok = 0
    for prem, hyp, want in CASES:
        r = grounding.classify(prem, hyp)
        top = max(r, key=r.get)
        flag = "OK " if top == want else "XX "
        ok += top == want
        print(f"{flag} want={want:13s} got={top:13s}  "
              f"e={r['entailment']:.3f} n={r['neutral']:.3f} c={r['contradiction']:.3f}")
        print(f"      premise={prem!r}")
        print(f"      hypoth.={hyp!r}")

    print(f"\nclass-agreement: {ok}/{len(CASES)}")

    # helper-direction checks (the thing wiring depends on)
    print("\n--- helper directions ---")
    c_conflict = grounding.contradiction_prob(
        "The user is vegan and avoids all animal products.", "The user loves steak.")
    c_agree = grounding.contradiction_prob(
        "The user is a freelance fashion designer.", "The user works in fashion.")
    e_support = grounding.entailment_prob(
        "The user lives in Berlin, Germany.", "The user lives in Germany.")
    e_conflict = grounding.entailment_prob(
        "The user is allergic to peanuts.", "The user can safely eat peanuts.")
    print(f"contradiction_prob  conflict={c_conflict:.3f}  (want HIGH)   "
          f"agreement={c_agree:.3f}  (want LOW)")
    print(f"entailment_prob     support ={e_support:.3f}  (want HIGH)   "
          f"conflict ={e_conflict:.3f}  (want LOW)")
    dir_ok = (c_conflict > 0.5 and c_agree < 0.5 and e_support > 0.5 and e_conflict < 0.5)
    print(f"helper directions correct: {dir_ok}")

    # idempotent prewarm / cached instance
    grounding.prewarm()
    m2, _ = grounding.get_model()
    print(f"\ncached-instance identity (prewarm idempotent): {m2 is m}")


if __name__ == "__main__":
    main()
