"""Spec for the continuous batching scheduler.

The scheduler owns the engine thread. Requests submitted from asyncio land in
the engine at step boundaries (continuous admission), stream their tokens back
as they generate, and can be cancelled mid-flight.
"""
import asyncio

import pytest

from fake_serving import FakeEngine
from serving.scheduler import Scheduler, SchedulerFull


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=15))


async def consume(handle):
    tokens = []
    finish = None
    async for event in handle:
        if event.token is not None:
            tokens.append(event.token)
        if event.finish_reason is not None:
            finish = event.finish_reason
    return tokens, finish


def test_single_request_streams_scripted_tokens():
    async def go():
        engine = FakeEngine()
        sched = Scheduler(engine)
        sched.start()
        try:
            handle = await sched.submit([100], max_tokens=5)
            tokens, finish = await consume(handle)
            assert tokens == [101, 102, 103, 104, 105]
            assert finish == "length"
        finally:
            sched.stop()
    run(go())


def test_stop_token_is_not_surfaced():
    async def go():
        engine = FakeEngine(eos_token=103)
        sched = Scheduler(engine)
        sched.start()
        try:
            handle = await sched.submit([100], max_tokens=50)
            tokens, finish = await consume(handle)
            assert tokens == [101, 102], "the stop token itself must be suppressed"
            assert finish == "stop"
        finally:
            sched.stop()
    run(go())


def test_concurrent_requests_do_not_cross_talk():
    async def go():
        engine = FakeEngine()
        sched = Scheduler(engine)
        sched.start()
        try:
            handles = [await sched.submit([base], max_tokens=6) for base in (100, 200, 300, 400)]
            results = await asyncio.gather(*(consume(h) for h in handles))
            for base, (tokens, finish) in zip((100, 200, 300, 400), results):
                assert tokens == [base + i for i in range(1, 7)]
                assert finish == "length"
        finally:
            sched.stop()
    run(go())


def test_late_request_is_admitted_while_first_still_running():
    async def go():
        engine = FakeEngine()
        sched = Scheduler(engine)
        sched.start()
        try:
            long_handle = await sched.submit([100], max_tokens=400)
            long_iter = long_handle.__aiter__()
            # let the long request produce some tokens first
            for _ in range(3):
                await long_iter.__anext__()
            short_handle = await sched.submit([500], max_tokens=3)
            first_short = await asyncio.wait_for(short_handle.__aiter__().__anext__(), 10)
            assert first_short.token == 501, "second request must stream before the first finishes"
            await long_handle.cancel()
        finally:
            sched.stop()
    run(go())


def test_tokens_arrive_in_generation_order():
    async def go():
        engine = FakeEngine()
        sched = Scheduler(engine)
        sched.start()
        try:
            handle = await sched.submit([1000], max_tokens=32)
            tokens, _ = await consume(handle)
            assert tokens == sorted(tokens)
            assert len(tokens) == 32
        finally:
            sched.stop()
    run(go())


def test_cancellation_removes_sequence_from_engine():
    async def go():
        engine = FakeEngine()
        sched = Scheduler(engine)
        sched.start()
        try:
            handle = await sched.submit([100], max_tokens=10**9)
            it = handle.__aiter__()
            for _ in range(3):
                await it.__anext__()
            await handle.cancel()
            for _ in range(50):
                if engine.removed:
                    break
                await asyncio.sleep(0.02)
            assert engine.removed, "engine.remove must be called on cancellation"
        finally:
            sched.stop()
    run(go())


def test_full_scheduler_rejects_submissions():
    async def go():
        engine = FakeEngine()
        sched = Scheduler(engine, max_pending=2)
        sched.start()
        try:
            a = await sched.submit([100], max_tokens=10**9)
            b = await sched.submit([200], max_tokens=10**9)
            with pytest.raises(SchedulerFull):
                await sched.submit([300], max_tokens=5)
            await a.cancel()
            await b.cancel()
        finally:
            sched.stop()
    run(go())


def test_per_request_state_machine_reaches_engine():
    async def go():
        engine = FakeEngine()
        sched = Scheduler(engine)
        sched.start()
        try:
            machine = object()
            with_machine = await sched.submit([100], max_tokens=2, state_machine=machine)
            plain = await sched.submit([200], max_tokens=2)
            await asyncio.gather(consume(with_machine), consume(plain))
            by_uid = {uid: sm for uid, sm in engine.inserted_state_machines}
            assert machine in by_uid.values(), "custom state machine must reach the engine"
            assert None in by_uid.values(), "plain requests must not get a machine"
        finally:
            sched.stop()
    run(go())


def test_per_request_sampler_reaches_engine():
    async def go():
        engine = FakeEngine()
        sched = Scheduler(engine)
        sched.start()
        try:
            marker = object()
            handle = await sched.submit([100], max_tokens=2, sampler=marker)
            await consume(handle)
            assert any(s is marker for _, s in engine.inserted_samplers)
        finally:
            sched.stop()
    run(go())


def test_stop_shuts_down_cleanly():
    async def go():
        engine = FakeEngine()
        sched = Scheduler(engine)
        sched.start()
        handle = await sched.submit([100], max_tokens=3)
        await consume(handle)
        sched.stop()
        assert not sched.is_running
    run(go())
