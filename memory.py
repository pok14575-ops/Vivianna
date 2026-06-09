import numpy as np
import pickle
import os
import threading
import time

try:
    from fastembed import TextEmbedding as _TextEmbedding
    _FASTEMBED_OK = True
except ImportError:
    _FASTEMBED_OK = False

from config import (
    MEMORY_VECTORS_PATH, MEMORY_META_PATH, MEMORY_TOP_K, MEMORY_EMBED_MODEL,
    MEMORY_EMBED_CACHE, SALIENCE_RANK_WEIGHT,
    MEMORY_RERANK_ENABLED, MEMORY_RERANK_TOP_N, CROSS_ENCODER_MODEL, CROSS_ENCODER_DEVICE,
    MEMORY_DEDUP_GUARD, MEMORY_DEDUP_THRESHOLD,
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

    # ── timestamps (foundation for memory age; see vivianna_memory_timestamps_plan) ──
    @staticmethod
    def _with_timestamp(metadata, created_at=None):
        """Return a metadata dict carrying a `created_at` (epoch seconds, float). Stamped at
        every write so memory age is computable (the time-gated contradiction clarify flow
        needs "age >= 12h"). setdefault: an explicit created_at (e.g. a merge preserving the
        oldest) is kept; otherwise stamp now. Legacy entries pre-dating this carry NO
        created_at — deliberately NOT backfilled (stamping them now would falsely make old
        facts look fresh); consumers treat a missing created_at as old/unknown."""
        md = dict(metadata or {})
        md.setdefault("created_at", created_at if created_at is not None else time.time())
        return md

    def _oldest_created_at(self, indices):
        """Min created_at among the given entry indices, so a fact's true age survives a
        merge (consolidating an old fact must NOT reset its age to now). None if none of
        them carry a timestamp. Caller must already hold self._lock."""
        stamps = [(self._memories[i].get("metadata") or {}).get("created_at")
                  for i in indices]
        stamps = [s for s in stamps if isinstance(s, (int, float))]
        return min(stamps) if stamps else None

    def add(self, text, metadata=None, dedup=False):
        """Append a memory. Returns True if stored, False if skipped.

        dedup=True (auto-save path): if `text` is a near-identical duplicate of an
        existing memory (cosine >= MEMORY_DEDUP_THRESHOLD), skip the append so recall
        turns re-extracting a known fact don't accumulate. Manual saves and audit
        merges pass dedup=False (never gated)."""
        if not self._available:
            return False
        with self._lock:
            vec = self._embed(text)
            if (dedup and MEMORY_DEDUP_GUARD and self._vectors is not None
                    and len(self._memories) > 0):
                sims = self._vectors @ vec
                j = int(np.argmax(sims))
                if float(sims[j]) >= MEMORY_DEDUP_THRESHOLD:
                    print(f"[MEMORY] Dedup skip (cosine={float(sims[j]):.3f} vs "
                          f"existing): {text[:60]}", flush=True)
                    return False
            if self._vectors is None:
                self._vectors = vec.reshape(1, -1)
                self._dim = vec.shape[0]
            else:
                self._vectors = np.vstack([self._vectors, vec])
            self._memories.append({"text": text, "metadata": self._with_timestamp(metadata)})
            self._save_unlocked()
            return True

    # ── audit support (used by brain.audit_scan / apply_audit) ───────────────
    def all_entries(self):
        """Snapshot of the store for auditing: [{index, text, metadata}, ...]."""
        with self._lock:
            return [{"index": i, "text": m["text"],
                     "metadata": dict(m.get("metadata") or {})}
                    for i, m in enumerate(self._memories)]

    def cluster_candidates(self, cos_threshold):
        """Group entries into near-duplicate clusters by cosine connected-components.
        Returns a list of index-lists (each length >= 2); singletons omitted."""
        with self._lock:
            n = len(self._memories)
            if self._vectors is None or n < 2:
                return []
            S = self._vectors @ self._vectors.T
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for i in range(n):
                for k in range(i + 1, n):
                    if S[i, k] >= cos_threshold:
                        parent[find(i)] = find(k)
            groups = {}
            for i in range(n):
                groups.setdefault(find(i), []).append(i)
            return [sorted(g) for g in groups.values() if len(g) >= 2]

    def apply_merge(self, indices, new_text, metadata=None):
        """Remove `indices` and append one merged entry. Returns the new count.
        Recomputes vectors by slicing (no re-embed of survivors)."""
        with self._lock:
            oldest = self._oldest_created_at(indices)   # before we drop them — preserve fact age
            drop = set(indices)
            keep = [i for i in range(len(self._memories)) if i not in drop]
            self._memories = [self._memories[i] for i in keep]
            self._vectors = (self._vectors[keep] if self._vectors is not None
                             and len(keep) > 0 else None)
            vec = self._embed(new_text)
            if self._vectors is None or len(self._vectors) == 0:
                self._vectors = vec.reshape(1, -1)
                self._dim = vec.shape[0]
            else:
                self._vectors = np.vstack([self._vectors, vec])
            self._memories.append({"text": new_text,
                                   "metadata": self._with_timestamp(
                                       metadata or {"source": "audit-merge"}, oldest)})
            self._save_unlocked()
            return len(self._memories)

    def delete_indices(self, indices):
        """Remove the given indices. Returns the new count."""
        with self._lock:
            drop = set(indices)
            keep = [i for i in range(len(self._memories)) if i not in drop]
            self._memories = [self._memories[i] for i in keep]
            self._vectors = (self._vectors[keep] if self._vectors is not None
                             and len(keep) > 0 else None)
            self._save_unlocked()
            return len(self._memories)

    def apply_audit_batch(self, merges, deletes=None):
        """Apply approved audit actions in ONE pass (index-safe — clusters are
        disjoint, so all removals reference the ORIGINAL indices). `merges` is a list
        of (indices, merged_text); `deletes` a flat index list. Removes the union of
        all referenced indices, then appends one entry per merged_text. Returns the
        new count."""
        with self._lock:
            drop = set(deletes or [])
            appends = []   # (text, created_at) — created_at = oldest of the merged entries
            for idxs, text in (merges or []):
                drop.update(idxs)
                appends.append((text, self._oldest_created_at(idxs)))   # before reslice
            keep = [i for i in range(len(self._memories)) if i not in drop]
            new_mem = [self._memories[i] for i in keep]
            new_vecs = (self._vectors[keep] if self._vectors is not None
                        and len(keep) > 0 else None)
            for text, created in appends:
                vec = self._embed(text)
                if new_vecs is None or len(new_vecs) == 0:
                    new_vecs = vec.reshape(1, -1)
                    self._dim = vec.shape[0]
                else:
                    new_vecs = np.vstack([new_vecs, vec])
                new_mem.append({"text": text, "metadata": self._with_timestamp(
                    {"source": "audit-merge"}, created)})
            self._memories = new_mem
            self._vectors = new_vecs
            self._save_unlocked()
            return len(self._memories)

    # ── contradiction clarify-and-resolve primitives (vivianna_contradiction_clarify_plan) ──
    # All three identify the target by EXACT stored text (the clarify flow parks the
    # verbatim memory string from query(), so it matches; auto-save appends never shift an
    # existing entry, and dedup keeps texts unique). Each is self-contained under one lock.
    def _index_of_text(self, text):
        """Index of the first entry whose text == `text`, else None. Caller holds the lock."""
        for i, m in enumerate(self._memories):
            if m["text"] == text:
                return i
        return None

    def reconfirm(self, text):
        """'Keep + re-stamp' resolution: the user re-confirmed a contradicted fact is still
        true, so stamp `confirmed_at = now` WITHOUT touching the text or `created_at`. The
        clarify age gate reads `confirmed_at or created_at`, so this resets the 12h clock and
        the same fact won't be queried again for another window. Returns True if found."""
        with self._lock:
            i = self._index_of_text(text)
            if i is None:
                return False
            md = dict(self._memories[i].get("metadata") or {})
            md["confirmed_at"] = time.time()
            self._memories[i] = {**self._memories[i], "metadata": md}
            self._save_unlocked()
            return True

    def replace(self, old_text, new_text, metadata=None):
        """'Update' resolution: the user gave a changed value. Drop the old entry and append
        the new fact with a FRESH created_at (the fact genuinely changed now, so it should not
        be eligible for re-clarify until it too ages). Returns True if the old entry existed."""
        with self._lock:
            i = self._index_of_text(old_text)
            if i is None:
                return False
            keep = [j for j in range(len(self._memories)) if j != i]
            self._memories = [self._memories[j] for j in keep]
            self._vectors = (self._vectors[keep] if self._vectors is not None
                             and len(keep) > 0 else None)
            vec = self._embed(new_text)
            if self._vectors is None or len(self._vectors) == 0:
                self._vectors = vec.reshape(1, -1)
                self._dim = vec.shape[0]
            else:
                self._vectors = np.vstack([self._vectors, vec])
            self._memories.append({"text": new_text, "metadata": self._with_timestamp(
                metadata or {"source": "clarify-update"})})
            self._save_unlocked()
            return True

    def delete_by_text(self, text):
        """'Delete' resolution: bare negation, no replacement. Returns True if found."""
        with self._lock:
            i = self._index_of_text(text)
            if i is None:
                return False
            keep = [j for j in range(len(self._memories)) if j != i]
            self._memories = [self._memories[j] for j in keep]
            self._vectors = (self._vectors[keep] if self._vectors is not None
                             and len(keep) > 0 else None)
            self._save_unlocked()
            return True

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
            # Second stage: cross-encoder rerank of the top-N cosine candidates.
            # Only the ORDER / membership of the returned top-k changes; "score"
            # stays the raw cosine so downstream confidence logic is unchanged.
            if MEMORY_RERANK_ENABLED and len(scores) > 1:
                reranked = self._rerank(text, ranking, k)
                if reranked is not None:
                    return [{**self._memories[i], "score": float(scores[i])} for i in reranked]
            indices = np.argsort(ranking)[-k:][::-1]
            return [{**self._memories[i], "score": float(scores[i])} for i in indices]

    def _rerank(self, text, ranking, k):
        """Reorder the top-N cosine candidates with the shared cross-encoder.
        Returns a list of corpus indices (best-first, length<=k), or None to fall
        back to cosine order (model unavailable / any failure)."""
        try:
            from cross_encoder_model import get_model
            model = get_model(CROSS_ENCODER_MODEL, CROSS_ENCODER_DEVICE)
            if model is None:
                return None
            n = min(MEMORY_RERANK_TOP_N, len(ranking))
            cand = list(np.argsort(ranking)[-n:][::-1])      # top-N by cosine(+salience)
            ce = model.predict([[text, self._memories[i]["text"]] for i in cand],
                               show_progress_bar=False)
            order = [cand[j] for j in np.argsort(ce)[::-1]]
            return order[:k]
        except Exception as e:  # noqa: BLE001
            print(f"[MEMORY] rerank failed ({type(e).__name__}: {e}); cosine order.", flush=True)
            return None

    def context_block(self, text):
        results = self.query(text)
        if not results:
            return ""
        # Bridging instruction: memories are STORED third-person ("The user…") so the
        # auto-save/dedup/audit paths stay consistent (see encoder-stack work), but the 9B
        # is talking TO that person — without this line it can parrot the stored voice back
        # ("they prefer winter") instead of addressing Jamie directly ("you prefer winter").
        # Found via Qwen model-adjacency audit 2026-06-07.
        lines = [
            '[Note: The following facts are about the person you are talking to. '
            'Refer to them as "you" in your response.]',
            "[Long-term memories — use only if relevant to this conversation:]",
        ]
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
