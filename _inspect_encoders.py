import json, os
from huggingface_hub import snapshot_download

nli = snapshot_download("tasksource/deberta-small-long-nli")
emo = snapshot_download("SamLowe/roberta-base-go_emotions")

def cfg(p):
    with open(os.path.join(p, "config.json"), encoding="utf-8") as f:
        return json.load(f)

n = cfg(nli)
print("=== deberta-small-long-nli ===")
print("id2label:", n.get("id2label"))
print("max_position_embeddings:", n.get("max_position_embeddings"))
id2 = {int(k): v.lower() for k, v in n.get("id2label", {}).items()}
exp = {0: "entailment", 1: "neutral", 2: "contradiction"}
print("NLI ORDER == e/n/c (0/1/2):", id2 == exp)
print("actual order:", id2)

e = cfg(emo)
lbls = e.get("id2label", {})
print()
print("=== roberta-base-go_emotions ===")
print("num_labels:", len(lbls))
print("problem_type:", e.get("problem_type"))
print("labels:", [lbls[str(i)] for i in range(len(lbls))])

def dirsize(p):
    root = os.path.dirname(os.path.dirname(p))  # the models--... dir
    t = 0
    for r, _, fs in os.walk(root):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return round(t / 1e6, 1)

print()
print("=== disk (MB, incl blobs) ===")
print("nli :", dirsize(nli))
print("emo :", dirsize(emo))
