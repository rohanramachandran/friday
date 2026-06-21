"""Working-context management and compaction, with the embedder faked out."""
import numpy as np
import pytest

import orchestrator.memory as memory_mod
from orchestrator.memory import Memory


class FakeEmbedder:
    """Deterministic embeddings: one hot-ish vectors keyed by text hash."""

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.zeros(8)
            v[hash(t) % 8] = 1.0
            out.append(v)
        return np.array(out)


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_mod, "DATA", tmp_path)
    m = Memory()
    m.log_path = tmp_path / "conversation.jsonl"
    monkeypatch.setattr(m, "_embedder_lazy", lambda: FakeEmbedder())
    return m


def test_turns_append_in_order(mem):
    mem.add_user("hello")
    mem.add_assistant("hi")
    msgs = mem.context_messages()
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[0]["content"] == "hello"


def test_conversation_is_logged_to_disk(mem):
    mem.add_user("logged?")
    lines = mem.log_path.read_text().strip().splitlines()
    assert len(lines) == 1


def test_summary_prepended_when_present(mem):
    mem.summary = "earlier stuff"
    mem.add_user("now")
    msgs = mem.context_messages()
    assert msgs[0]["role"] == "system"
    assert "earlier stuff" in msgs[0]["content"]


def test_compaction_halves_working_and_builds_index(mem, monkeypatch):
    monkeypatch.setattr(memory_mod, "COMPACT_TRIGGER", 10)
    for i in range(6):
        mem.add_user(f"message number {i} with some padding text")
    assert len(mem.working) < 6
    assert mem.summary
    assert mem._index is not None
    assert len(mem._docs) == len(mem._index)


def test_search_returns_indexed_docs(mem, monkeypatch):
    monkeypatch.setattr(memory_mod, "COMPACT_TRIGGER", 10)
    for i in range(6):
        mem.add_user(f"message number {i} with some padding text")
    results = mem.search("message number 0 with some padding text", k=2)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)


def test_search_empty_index_returns_nothing(mem):
    assert mem.search("anything") == []
