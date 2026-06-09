import time, torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, pipeline)

dev = 0 if torch.cuda.is_available() else -1
print(f"device: {'cuda:0' if dev==0 else 'cpu'}  torch={torch.__version__}")
print()

# ---------- NLI: raw 3-class, the grounding/contradiction path ----------
NLI = "tasksource/deberta-small-long-nli"
tok = AutoTokenizer.from_pretrained(NLI)
mdl = AutoModelForSequenceClassification.from_pretrained(NLI)
if dev == 0:
    mdl = mdl.to("cuda")
mdl.eval()
id2 = mdl.config.id2label

def nli(premise, hypothesis):
    x = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=1680)
    if dev == 0:
        x = {k: v.to("cuda") for k, v in x.items()}
    with torch.no_grad():
        probs = mdl(**x).logits.softmax(-1)[0].tolist()
    return {id2[i]: round(p, 3) for i, p in enumerate(probs)}

print("=== NLI raw 3-class (grounding path) ===")
print("ENTAIL case   premise='The cat sat on the mat.' hyp='There is a cat on the mat.'")
print("  ->", nli("The cat sat on the mat.", "There is a cat on the mat."))
print("CONTRA case   premise='The store is open today.' hyp='The store is closed today.'")
print("  ->", nli("The store is open today.", "The store is closed today."))
print("NEUTRAL case  premise='She bought some bread.' hyp='She is allergic to gluten.'")
print("  ->", nli("She bought some bread.", "She is allergic to gluten."))
print()

# ---------- NLI: zero-shot wrapper, the salience/utility path ----------
print("=== NLI zero-shot wrapper (salience/utility path) ===")
zs = pipeline("zero-shot-classification", model=NLI, device=dev)
r = zs("Remember my wife is allergic to peanuts.",
       candidate_labels=["a fact worth remembering long-term", "small talk", "a question"])
print("salience probe ->", {l: round(s, 3) for l, s in zip(r["labels"], r["scores"])})
print()

# ---------- Emotion: 28-way multi-label sigmoid ----------
print("=== go_emotions multi-label (top activations) ===")
emo = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions",
               top_k=None, device=dev)
for s in ["Thank you so much, this honestly made my day!",
          "I'm really worried this whole thing is going to fall apart."]:
    out = sorted(emo(s)[0], key=lambda d: d["score"], reverse=True)[:4]
    print(f"  {s!r}")
    print("    ->", {d["label"]: round(d["score"], 3) for d in out})
print()

# ---------- VRAM ----------
if dev == 0:
    torch.cuda.synchronize()
    print(f"peak VRAM both encoders: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
