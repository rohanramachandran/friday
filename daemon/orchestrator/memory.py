"""Hierarchical memory: working context + long-term FAISS store."""
import json, time
from pathlib import Path
from typing import List, Dict
import numpy as np

WORKING_TOKEN_BUDGET = 8000
COMPACT_TRIGGER = 24000

DATA = Path.home() / ".friday"
DATA.mkdir(exist_ok=True)

class Memory:
    def __init__(self):
        self.working: List[Dict] = []  # recent turns verbatim
        self.summary: str = ""          # compressed older context
        self.log_path = DATA / "conversation.jsonl"
        self._embedder = None
        self._index = None
        self._docs: List[str] = []
        self._load()

    def _load(self):
        idx_path = DATA / "index.npy"
        docs_path = DATA / "docs.json"
        if idx_path.exists() and docs_path.exists():
            self._index = np.load(idx_path)
            self._docs = json.loads(docs_path.read_text())

    def _embedder_lazy(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
        return self._embedder

    def add_user(self, text: str):
        self.working.append({"role": "user", "content": text})
        self._log({"role": "user", "content": text, "ts": time.time()})
        self._maybe_compact()

    def add_assistant(self, text: str):
        self.working.append({"role": "assistant", "content": text})
        self._log({"role": "assistant", "content": text, "ts": time.time()})

    def _log(self, entry):
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")

    def context_messages(self) -> List[Dict]:
        msgs = []
        if self.summary:
            msgs.append({"role": "system", "content": f"[Earlier conversation summary: {self.summary}]"})
        msgs.extend(self.working)
        return msgs

    def _estimate_tokens(self) -> int:
        return sum(len(m["content"]) for m in self.working) // 4

    def _maybe_compact(self):
        if self._estimate_tokens() < COMPACT_TRIGGER:
            return
        # compact oldest half into summary, embed the raw text for retrieval
        half = len(self.working) // 2
        old = self.working[:half]
        self.working = self.working[half:]
        old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old)
        # naive summary; replace with a model call later if desired
        new_summary = f"{self.summary}\n[Older turns]: {old_text[:1500]}"[-2000:]
        self.summary = new_summary
        # embed old turns for retrieval
        try:
            emb = self._embedder_lazy().encode([m["content"] for m in old])
            if self._index is None:
                self._index = emb
            else:
                self._index = np.vstack([self._index, emb])
            self._docs.extend([m["content"] for m in old])
            np.save(DATA / "index.npy", self._index)
            (DATA / "docs.json").write_text(json.dumps(self._docs))
        except Exception:
            pass

    def search(self, query: str, k: int = 5) -> List[str]:
        if self._index is None or len(self._docs) == 0:
            return []
        try:
            q_emb = self._embedder_lazy().encode([query])[0]
            sims = self._index @ q_emb / (np.linalg.norm(self._index, axis=1) * np.linalg.norm(q_emb) + 1e-8)
            top = np.argsort(-sims)[:k]
            return [self._docs[i] for i in top]
        except Exception:
            return []
