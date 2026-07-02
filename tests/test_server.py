"""Spec for the /generate HTTP API."""
import asyncio
import json

import httpx
import pytest

from fake_serving import FakeEngine, FakeTokenizer
from serving.scheduler import Scheduler
from serving.server import create_app


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=15))


def make_client(engine=None, **sched_kwargs):
    engine = engine or FakeEngine()
    sched = Scheduler(engine, **sched_kwargs)
    sched.start()
    app = create_app(sched, FakeTokenizer(), sampler_factory=lambda temperature, top_p: None)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    return client, sched, engine


def parse_sse(raw: str):
    events = []
    for block in raw.split("\n\n"):
        block = block.strip()
        if block.startswith("data: "):
            events.append(json.loads(block[len("data: "):]))
    return events


def test_generate_non_streaming():
    async def go():
        client, sched, _ = make_client()
        try:
            # prompt "d" encodes to [100]; fake engine counts up from there
            r = await client.post("/generate", json={
                "prompt": "d", "max_tokens": 4, "stream": False, "template": False})
            assert r.status_code == 200
            body = r.json()
            assert body["text"] == "efgh"
            assert body["finish_reason"] == "length"
            assert body["prompt_tokens"] == 1
            assert body["completion_tokens"] == 4
        finally:
            await client.aclose()
            sched.stop()
    run(go())


def test_generate_streaming_sse():
    async def go():
        client, sched, _ = make_client()
        try:
            async with client.stream("POST", "/generate", json={
                    "prompt": "d", "max_tokens": 4, "stream": True, "template": False}) as r:
                assert r.status_code == 200
                assert r.headers["content-type"].startswith("text/event-stream")
                raw = ""
                async for chunk in r.aiter_text():
                    raw += chunk
            events = parse_sse(raw)
            deltas = [e["text"] for e in events if "text" in e]
            assert "".join(deltas) == "efgh"
            final = events[-1]
            assert final.get("done") is True
            assert final["finish_reason"] == "length"
            assert final["completion_tokens"] == 4
        finally:
            await client.aclose()
            sched.stop()
    run(go())


def test_concurrent_streams_are_isolated():
    async def go():
        client, sched, _ = make_client()
        try:
            async def fetch(ch):
                r = await client.post("/generate", json={
                    "prompt": ch, "max_tokens": 3, "stream": False, "template": False})
                return r.json()["text"]

            texts = await asyncio.gather(*(fetch(c) for c in "dnx"))
            assert texts == ["efg", "opq", "yz{"]
        finally:
            await client.aclose()
            sched.stop()
    run(go())


def test_busy_server_returns_503():
    async def go():
        # httpx's ASGITransport buffers full responses, so hold the slot by
        # submitting to the scheduler directly and hit the API while it is busy
        client, sched, _ = make_client(engine=FakeEngine(step_delay=0.005), max_pending=1)
        try:
            occupant = await sched.submit([100], max_tokens=8192)
            r = await client.post("/generate", json={
                "prompt": "n", "max_tokens": 2, "stream": False, "template": False})
            assert r.status_code == 503
            await occupant.cancel()
        finally:
            await client.aclose()
            sched.stop()
    run(go())


def test_empty_prompt_rejected():
    async def go():
        client, sched, _ = make_client()
        try:
            r = await client.post("/generate", json={"prompt": "", "template": False})
            assert r.status_code == 422
        finally:
            await client.aclose()
            sched.stop()
    run(go())


def test_ignore_eos_sends_no_stop_machine_to_engine():
    async def go():
        engine = FakeEngine()
        sched = Scheduler(engine)
        sched.start()
        marker = object()
        app = create_app(sched, FakeTokenizer(),
                         sampler_factory=lambda temperature, top_p: None,
                         no_stop_machine_factory=lambda: marker)
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://test")
        try:
            r = await client.post("/generate", json={
                "prompt": "d", "max_tokens": 3, "stream": False,
                "template": False, "ignore_eos": True})
            assert r.status_code == 200
            machines = [sm for _, sm in engine.inserted_state_machines]
            assert marker in machines
        finally:
            await client.aclose()
            sched.stop()
    run(go())


def test_health():
    async def go():
        client, sched, _ = make_client()
        try:
            r = await client.get("/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"
        finally:
            await client.aclose()
            sched.stop()
    run(go())
