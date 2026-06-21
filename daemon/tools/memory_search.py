"""Search past conversations via embedded memory."""
import json
from pathlib import Path
import numpy as np

DATA = Path.home() / ".friday"

_embedder = None

def _emb():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    return _embedder

async def search_tool(query: str, k: int = 5) -> str:
    idx_path = DATA / "index.npy"
    docs_path = DATA / "docs.json"
    if not idx_path.exists():
        return "No memory yet."
    index = np.load(idx_path)
    docs = json.loads(docs_path.read_text())
    q = _emb().encode([query])[0]
    sims = index @ q / (np.linalg.norm(index, axis=1) * np.linalg.norm(q) + 1e-8)
    top = np.argsort(-sims)[:k]
    results = [docs[i] for i in top]
    return "\n---\n".join(results) if results else "No matches."
