import numpy as np
import pickle
import os
import threading

try:
    from fastembed import TextEmbedding as _TextEmbedding
    _FASTEMBED_OK = True
except ImportError:
    _FASTEMBED_OK = False

from config import (
    MEMORY_VECTORS_PATH, MEMORY_META_PATH, MEMORY_TOP_K, MEMORY_EMBED_MODEL,
    MEMORY_EMBED_CACHE, SALIENCE_RANK_WEIGHT,
)


class MemoryManager:
    def __init__(self):
        self._lock = threading.Lock()
        if not _FASTEMBED_OK:
            print("[MEMORY] fastembed not installed — long-term memory disabled.", flush=True)
            self._available = False
            self._model = None
        else:
            try:
                os.makedirs(MEMORY_EMBED_CACHE, exist_ok=True)
                self._model = _TextEmbedding(MEMORY_EMBED_MODEL, cache_dir=MEMORY_EMBED_CACHE)
                self._available = True
            except Exception as e:
                print(f"[MEMORY] Embedding model failed to load: {e}", flush=True)
                self._available = False
                self._model = None

        self._dim = None
        self._vectors = None  # np.ndarray shape (N, dim) or None
        self._memories = []   # list of {"text": str, "metadata": dict}

        if self._available:
            self._load()

    # ------------------------------------------------------------------

    def _embed(self, text):
        vecs = list(self._model.embed([text]))
        v = np.array(vecs[0], dtype="float32")
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v

    def add(self, text, metadata=None):
        if not self._available:
            return
        with self._lock:
            vec = self._embed(text)
            if self._vectors is None:
                self._vectors = vec.reshape(1, -1)
                self._dim = vec.shape[0]
            else:
                self._vectors = np.vstack([self._vectors, vec])
            self._memories.append({"text": text, "metadata": metadata or {}})
            self._save_unlocked()

    def query(self, text, top_k=None, salience_weight=None):
        if not self._available or not self._memories:
            return []
        k = top_k or MEMORY_TOP_K
        w = SALIENCE_RANK_WEIGHT if salience_weight is None else salience_weight
        with self._lock:
            vec = self._embed(text)
            scores = self._vectors @ vec
            # Selection ranking may blend in stored salience; the returned "score"
            # always stays the raw cosine so downstream confidence logic is unchanged.
            if w > 0.0:
                sal = np.array([
                    float((self._memories[i].get("metadata") or {}).get("salience", 0.5))
                    for i in range(len(self._memories))
                ], dtype="float32")
                ranking = scores + w * sal
            else:
                ranking = scores
            k = min(k, len(scores))
            indices = np.argsort(ranking)[-k:][::-1]
            return [{**self._memories[i], "score": float(scores[i])} for i in indices]

    def context_block(self, text):
        results = self.query(text)
        if not results:
            return ""
        lines = ["[Long-term memories — use only if relevant to this conversation:]"]
        for r in results:
            lines.append(f"- {r['text']}")
        lines.append("")
        return "\n".join(lines)

    def purge(self):
        if not self._available:
            return
        with self._lock:
            self._vectors = None
            self._memories = []
            self._save_unlocked()
        print("[MEMORY] Purged all long-term memories.", flush=True)

    def count(self):
        return len(self._memories)

    # ------------------------------------------------------------------

    def _save_unlocked(self):
        os.makedirs(os.path.dirname(MEMORY_VECTORS_PATH), exist_ok=True)
        arr = self._vectors if self._vectors is not None else np.zeros((0,), dtype="float32")
        np.save(MEMORY_VECTORS_PATH, arr)
        with open(MEMORY_META_PATH, "wb") as f:
            pickle.dump(self._memories, f)

    def _load(self):
        if not (os.path.exists(MEMORY_VECTORS_PATH) and os.path.exists(MEMORY_META_PATH)):
            return
        try:
            arr = np.load(MEMORY_VECTORS_PATH, allow_pickle=False)
            with open(MEMORY_META_PATH, "rb") as f:
                mems = pickle.load(f)
            if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[0] == len(mems):
                self._vectors = arr
                self._memories = mems
                self._dim = arr.shape[1]
                print(f"[MEMORY] Loaded {len(mems)} long-term memories.", flush=True)
        except Exception as e:
            print(f"[MEMORY] Load error (starting fresh): {e}", flush=True)
