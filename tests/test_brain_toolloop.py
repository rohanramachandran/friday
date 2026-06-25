"""The tool loop must speak the final answer exactly once."""
import asyncio

import orchestrator.brain as brain_mod
from orchestrator.brain import Brain


class StubMemory:
    def __init__(self):
        self.assistant_turns = []

    def add_user(self, text):
        pass

    def add_assistant(self, text):
        self.assistant_turns.append(text)

    def context_messages(self):
        return []


def scripted_brain(scripts):
    """Brain whose _stream plays back one script per call instead of running a model."""
    b = Brain.__new__(Brain)
    b.memory = StubMemory()
    b.sampler = None
    b.model = None
    b.tokenizer = None
    calls = {"n": 0}

    async def fake_stream(prompt, max_tokens=512):
        script = scripts[calls["n"]]
        calls["n"] += 1
        for event in script:
            yield event

    b._stream = fake_stream
    b._build_prompt = lambda *a, **k: "PROMPT"
    b._calls = calls
    return b


def collect(brain, text="q"):
    async def go():
        return [e async for e in brain.run(text)]
    return asyncio.run(go())


def test_answer_after_tool_call_is_emitted_once(monkeypatch):
    async def fake_run_tool(name, args):
        return "4"
    monkeypatch.setattr(brain_mod, "run_tool", fake_run_tool)

    answer = "The answer is 4."
    b = scripted_brain([
        [("tool", '<tool>{"name":"run_code","args":{"code":"print(2+2)"}}</tool>')],
        [("delta", answer), ("end", answer)],
    ])
    events = collect(b)

    spoken = "".join(e["text"] for e in events if e["type"] == "token")
    assert spoken == answer
    assert b._calls["n"] == 2, "phase 2 must not regenerate an answer phase 1 already gave"
    assert b.memory.assistant_turns == [answer]
    assert events[-1]["type"] == "done"


def test_forced_summary_still_runs_when_phase1_stays_silent(monkeypatch):
    async def fake_run_tool(name, args):
        return "4"
    monkeypatch.setattr(brain_mod, "run_tool", fake_run_tool)

    b = scripted_brain([
        [("tool", '<tool>{"name":"run_code","args":{"code":"print(2+2)"}}</tool>')],
        [("end", "")],
        [("delta", "It is 4."), ("end", "It is 4.")],
    ])
    events = collect(b)

    spoken = "".join(e["text"] for e in events if e["type"] == "token")
    assert spoken == "It is 4."
    assert b._calls["n"] == 3, "phase 2 should run when phase 1 produced no text"


def test_no_tool_path_unchanged():
    b = scripted_brain([
        [("delta", "Hello."), ("end", "Hello.")],
    ])
    events = collect(b)
    spoken = "".join(e["text"] for e in events if e["type"] == "token")
    assert spoken == "Hello."
    assert b._calls["n"] == 1
