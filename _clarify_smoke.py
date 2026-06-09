r"""Offline smoke for the time-gated contradiction clarify-and-resolve flow.

No llama-server, no models. Three parts:
  A. memory.py resolution primitives (reconfirm / replace / delete_by_text) via __new__ +
     a fake _embed + no-disk _save (mirrors the timestamps test).
  B. stabilizer.pre_generate populates PreGenResult.conflict_texts (fake grounding module).
  C. brain.py's pure clarify logic via AST-extract + a controlled namespace (brain.py is too
     heavy to import — it builds the OpenAI client / MemoryManager / layers at module load).

Run:  venv_tf5\Scripts\python.exe _clarify_smoke.py
"""
import ast
import sys
import time
import types
import threading

import numpy as np

PASS, FAIL = [], []


def check(name, cond):
    (PASS if cond else FAIL).append(name)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


# ── Part A: memory.py primitives ────────────────────────────────────────────────
print("\nPart A — memory.py reconfirm / replace / delete_by_text")
from memory import MemoryManager

m = MemoryManager.__new__(MemoryManager)
m._lock = threading.Lock()
m._available = True
m._dim = 3
m._memories = []
m._vectors = None
m._save_unlocked = lambda: None
# deterministic, distinct unit-ish vectors per text
_counter = {"n": 0}
def _fake_embed(text):
    _counter["n"] += 1
    v = np.array([_counter["n"], len(text) % 5 + 1, 1.0], dtype="float32")
    return v / np.linalg.norm(v)
m._embed = _fake_embed

OLD = "The user prefers winter over summer due to a dislike of the heat."
m.add(OLD, metadata={"source": "auto", "created_at": 1000.0}, dedup=False)
m.add("The user is a freelance fashion designer.", metadata={"created_at": 2000.0}, dedup=False)
check("seeded 2 entries", m.count() == 2 and m._vectors.shape == (2, 3))

# reconfirm: stamps confirmed_at, leaves text + created_at intact
ok = m.reconfirm(OLD)
md = m._memories[0]["metadata"]
check("reconfirm found+returned True", ok is True)
check("reconfirm set confirmed_at", isinstance(md.get("confirmed_at"), float))
check("reconfirm kept created_at=1000", md.get("created_at") == 1000.0)
check("reconfirm kept text", m._memories[0]["text"] == OLD)
check("reconfirm missing -> False", m.reconfirm("nonexistent fact") is False)

# replace: drops old, appends new with FRESH created_at + clarify-update source
NEW = "The user now prefers summer and enjoys hot days."
n_before = m.count()
ok = m.replace(OLD, NEW)
texts = [e["text"] for e in m._memories]
check("replace returned True", ok is True)
check("replace removed old text", OLD not in texts)
check("replace added new text", NEW in texts)
check("replace kept count stable", m.count() == n_before)
new_md = m._memories[[e["text"] for e in m._memories].index(NEW)]["metadata"]
check("replace fresh created_at (not 1000)", new_md.get("created_at", 0) > 1000.0)
check("replace source=clarify-update", new_md.get("source") == "clarify-update")
check("replace vectors row count matches", m._vectors.shape[0] == m.count())
check("replace missing old -> False", m.replace("nope", "x") is False)

# delete_by_text
n_before = m.count()
ok = m.delete_by_text(NEW)
check("delete returned True", ok is True)
check("delete removed entry", NEW not in [e["text"] for e in m._memories])
check("delete decremented count", m.count() == n_before - 1)
check("delete vectors row count matches", (m._vectors.shape[0] if m._vectors is not None else 0) == m.count())
check("delete missing -> False", m.delete_by_text("gone already") is False)


# ── Part B: stabilizer conflict_texts ───────────────────────────────────────────
print("\nPart B — stabilizer.pre_generate -> conflict_texts")
# Inject a fake `grounding` module BEFORE importing the stabilizer path that uses it.
fake_grounding = types.ModuleType("grounding")
_CONTRA = {"The user dislikes the heat.": 0.92}   # this premise contradicts; others ~0
fake_grounding.contradiction_prob = lambda premise, hyp: _CONTRA.get(premise, 0.05)
sys.modules["grounding"] = fake_grounding

from vivianna_stabilizer import ViviannaStabilizer, StabilizerConfig

stab = ViviannaStabilizer(StabilizerConfig(debug=False, grounding_enabled=True,
                                           grounding_contradict_threshold=0.60))
hits = [
    {"text": "The user dislikes the heat.", "metadata": {"created_at": 1000.0}, "score": 0.8},
    {"text": "The user is a fashion designer.", "metadata": {"created_at": 1000.0}, "score": 0.7},
]
pre = stab.pre_generate("Honestly I love a hot summer day", hits)
check("conflict flagged", pre.conflict is True)
check("conflict_texts has the contradicted memory", pre.conflict_texts == ["The user dislikes the heat."])
check("memory_valid False on conflict", pre.memory_valid is False)

# grounding disabled -> no conflict_texts
stab_off = ViviannaStabilizer(StabilizerConfig(debug=False, grounding_enabled=False))
pre_off = stab_off.pre_generate("Honestly I love a hot summer day", hits)
check("grounding off -> empty conflict_texts", pre_off.conflict_texts == [])


# ── Part C: brain.py clarify logic via AST-extract ──────────────────────────────
print("\nPart C — brain.py _effective_age_hours / _maybe_arm_clarify / _classify / resolve")
SRC = open("brain.py", encoding="utf-8").read()
tree = ast.parse(SRC)
wanted = {"_effective_age_hours", "_maybe_arm_clarify", "_classify_clarify_answer",
          "resolve_clarify"}
ns = {"time": time, "float": float, "isinstance": isinstance, "print": print,
      "hasattr": hasattr, "getattr": getattr,
      "CLARIFY_ENABLED": True, "CLARIFY_MIN_AGE_HOURS": 12.0,
      "_pending_clarify": None}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        exec(compile(ast.Module([node], []), "brain.py", "exec"), ns)
check("extracted all 4 funcs", wanted.issubset(ns.keys()))

# _effective_age_hours
now = time.time()
check("age: confirmed_at preferred over created_at",
      abs(ns["_effective_age_hours"]({"created_at": now - 36000, "confirmed_at": now - 3600}) - 1.0) < 0.1)
check("age: falls back to created_at", abs(ns["_effective_age_hours"]({"created_at": now - 7200}) - 2.0) < 0.1)
check("age: no stamp -> None (legacy=old)", ns["_effective_age_hours"]({}) is None)

# _maybe_arm_clarify — fake PreGenResult-ish object
class P:
    def __init__(self, cts): self.conflict_texts = cts

old_hit = {"text": "The user dislikes the heat.", "metadata": {"created_at": now - 100000}}  # ~28h
fresh_hit = {"text": "The user is tired today.", "metadata": {"created_at": now - 600}}       # 10min
legacy_hit = {"text": "The user has no timestamp.", "metadata": {}}                            # unknown=old

ns["_pending_clarify"] = None
cue = ns["_maybe_arm_clarify"]("I love the heat", [old_hit], P(["The user dislikes the heat."]))
check("arm: old contradiction -> cue returned", bool(cue) and "dislikes the heat" in cue)
check("arm: old contradiction -> pending set", ns["_pending_clarify"] is not None
      and ns["_pending_clarify"]["memory_text"] == "The user dislikes the heat.")

ns["_pending_clarify"] = None
cue_fresh = ns["_maybe_arm_clarify"]("I'm energetic", [fresh_hit], P(["The user is tired today."]))
check("arm: fresh contradiction -> no cue", cue_fresh == "")
check("arm: fresh contradiction -> no pending", ns["_pending_clarify"] is None)

ns["_pending_clarify"] = None
cue_legacy = ns["_maybe_arm_clarify"]("x", [legacy_hit], P(["The user has no timestamp."]))
check("arm: legacy (no stamp) -> armed (treated old)", bool(cue_legacy) and ns["_pending_clarify"] is not None)

ns["_pending_clarify"] = None
cue_none = ns["_maybe_arm_clarify"]("x", [old_hit], P([]))   # no conflict
check("arm: no conflict -> no cue, no pending", cue_none == "" and ns["_pending_clarify"] is None)

# disabled flag
ns["CLARIFY_ENABLED"] = False
ns["_pending_clarify"] = None
check("arm: flag off -> no-op", ns["_maybe_arm_clarify"]("x", [old_hit],
      P(["The user dislikes the heat."])) == "" and ns["_pending_clarify"] is None)
ns["CLARIFY_ENABLED"] = True

# _classify_clarify_answer — fake _llm_bare returning canned model output
for out, exp_action, exp_new in [
    ("KEEP", "KEEP", None),
    ("DELETE", "DELETE", None),
    ("UPDATE: The user now prefers summer.", "UPDATE", "The user now prefers summer."),
    ("UNRELATED", "UNRELATED", None),
    ("UPDATE:", "KEEP", None),            # malformed update -> safe keep
    ("garbled nonsense", "UNRELATED", None),
]:
    ns["_llm_bare"] = lambda prompt, max_tok=60, _o=out: _o
    a, nt = ns["_classify_clarify_answer"]("reply", "The user dislikes the heat.")
    check(f"classify '{out[:18]}' -> {exp_action}", a == exp_action and nt == exp_new)

# resolve_clarify — fake _memory, _classify, _respond_clarify_ack; check routing + backup
class FakeMem:
    def __init__(self): self.calls = []
    def reconfirm(self, t): self.calls.append(("reconfirm", t)); return True
    def delete_by_text(self, t): self.calls.append(("delete", t)); return True
    def replace(self, o, n): self.calls.append(("replace", o, n)); return True

def run_resolve(action, new_text, user_input="reply"):
    ns["_pending_clarify"] = {"memory_text": "The user dislikes the heat.", "user_claim": "x"}
    fm = FakeMem(); ns["_memory"] = fm
    acks = []; ns["_respond_clarify_ack"] = lambda *a: acks.append(a)
    backups = []; bk = lambda: backups.append(1)
    ns["_classify_clarify_answer"] = lambda ui, ot, _a=action, _n=new_text: (_a, _n)
    consumed = ns["resolve_clarify"](user_input, bk)
    return consumed, fm.calls, acks, backups

c, calls, acks, backups = run_resolve("KEEP", None)
check("resolve KEEP: consumed+reconfirm+backup+ack+cleared",
      c is True and calls == [("reconfirm", "The user dislikes the heat.")]
      and len(backups) == 1 and len(acks) == 1 and ns["_pending_clarify"] is None)

c, calls, _, backups = run_resolve("DELETE", None)
check("resolve DELETE: delete_by_text + backup", c is True and calls[0][0] == "delete" and len(backups) == 1)

c, calls, _, backups = run_resolve("UPDATE", "The user loves heat now.")
check("resolve UPDATE: replace(old,new) + backup",
      c is True and calls == [("replace", "The user dislikes the heat.", "The user loves heat now.")]
      and len(backups) == 1)

# UNRELATED -> not consumed, pending cleared, NO backup/mutation
c, calls, acks, backups = run_resolve("UNRELATED", None)
check("resolve UNRELATED: not consumed, no mutation, no backup, cleared",
      c is False and calls == [] and backups == [] and ns["_pending_clarify"] is None)

# slash-command / exit -> not consumed, pending KEPT, no LLM/backup
ns["_pending_clarify"] = {"memory_text": "X", "user_claim": "x"}
ns["_memory"] = FakeMem()
ns["_classify_clarify_answer"] = lambda *a: (_ for _ in ()).throw(AssertionError("should not classify a command"))
ns["_respond_clarify_ack"] = lambda *a: None
check("resolve '/audit': not consumed, pending kept",
      ns["resolve_clarify"]("/audit", lambda: None) is False and ns["_pending_clarify"] is not None)
check("resolve 'exit': not consumed, pending kept",
      ns["resolve_clarify"]("exit", lambda: None) is False and ns["_pending_clarify"] is not None)

# no pending -> False
ns["_pending_clarify"] = None
check("resolve with no pending -> False", ns["resolve_clarify"]("anything", lambda: None) is False)


# ── summary ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*50}\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("FAILURES:", FAIL)
    sys.exit(1)
print("ALL GREEN")
