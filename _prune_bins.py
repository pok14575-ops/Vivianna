import os, glob
from huggingface_hub import snapshot_download

before = 0
roots = []
for repo in ["tasksource/deberta-small-long-nli", "SamLowe/roberta-base-go_emotions"]:
    snap = snapshot_download(repo)
    models_dir = os.path.dirname(os.path.dirname(snap))  # models--... root
    roots.append(models_dir)

def dirsize(p):
    t = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            try: t += os.path.getsize(os.path.join(r, f))
            except OSError: pass
    return t

before = sum(dirsize(r) for r in roots)

# Find every pytorch_model.bin (snapshot entries) and resolve real targets (blob + snapshot copy)
removed = []
for root in roots:
    for path in glob.glob(os.path.join(root, "**", "pytorch_model.bin"), recursive=True):
        real = os.path.realpath(path)
        for target in {path, real}:
            if os.path.exists(target):
                sz = os.path.getsize(target)
                os.remove(target)
                removed.append((target, sz))

after = sum(dirsize(r) for r in roots)
print("REMOVED:")
for t, sz in removed:
    print(f"  {round(sz/1e6,1):>7} MB  {t}")
print(f"before: {round(before/1e6,1)} MB")
print(f"after : {round(after/1e6,1)} MB")
print(f"freed : {round((before-after)/1e6,1)} MB")
