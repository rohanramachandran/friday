"""Continuous batching scheduler.

Bridges asyncio request handlers to the MLX engine, which is synchronous and
must run on a single thread. The engine thread loops over decode steps; at
every step boundary it admits newly submitted requests into the running batch
(continuous batching) and applies cancellations. Tokens are handed back to
each request's event loop as they are generated.

The engine is any object with the mlx_lm BatchGenerator contract:
insert(prompts, max_tokens, samplers) -> uids, next_generated() -> responses
with (uid, token, finish_reason), remove(uids), close(). Tests inject a fake.
"""
import asyncio
import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, List, Optional

log = logging.getLogger("serving.scheduler")

IDLE_POLL_S = 0.05


class SchedulerFull(Exception):
    """Raised when the scheduler is at max_pending live requests."""


@dataclass
class TokenEvent:
    token: Optional[int]        # None for a suppressed stop token
    finish_reason: Optional[str]  # None mid-stream, else "stop" / "length"


class _End:
    pass


_END = _End()


class _Request:
    __slots__ = ("tokens", "max_tokens", "sampler", "state_machine", "queue", "loop", "uid", "dead", "finished")

    def __init__(self, tokens: List[int], max_tokens: int, sampler: Any, loop, state_machine: Any = None):
        self.tokens = tokens
        self.max_tokens = max_tokens
        self.sampler = sampler
        self.state_machine = state_machine
        self.loop = loop
        self.queue: asyncio.Queue = asyncio.Queue()
        self.uid: Optional[int] = None
        self.dead = False       # cancelled before admission
        self.finished = False


class StreamHandle:
    """Async iterator over one request's TokenEvents."""

    def __init__(self, request: _Request, scheduler: "Scheduler"):
        self._request = request
        self._scheduler = scheduler
        self._exhausted = False

    def __aiter__(self):
        return self

    async def __anext__(self) -> TokenEvent:
        if self._exhausted:
            raise StopAsyncIteration
        event = await self._request.queue.get()
        if isinstance(event, _End):
            self._exhausted = True
            raise StopAsyncIteration
        return event

    async def cancel(self):
        self._scheduler._cancel(self._request)


class Scheduler:
    def __init__(self, engine, *, max_pending: int = 32):
        self.engine = engine
        self.max_pending = max_pending
        self._pending: "queue.Queue[_Request]" = queue.Queue()
        self._active = {}  # uid -> _Request, engine thread only
        self._cancels: List[_Request] = []
        self._lock = threading.Lock()
        self._live = 0
        self._stop_flag = False
        self._thread: Optional[threading.Thread] = None

    # ---- asyncio side ----

    async def submit(self, prompt_tokens: List[int], max_tokens: int = 256, sampler=None,
                     state_machine=None) -> StreamHandle:
        with self._lock:
            if self._live >= self.max_pending:
                raise SchedulerFull(f"{self._live} live requests, limit {self.max_pending}")
            self._live += 1
        request = _Request(list(prompt_tokens), max_tokens, sampler,
                           asyncio.get_running_loop(), state_machine)
        self._pending.put(request)
        return StreamHandle(request, self)

    def _cancel(self, request: _Request):
        with self._lock:
            self._cancels.append(request)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def pending_count(self) -> int:
        return self._pending.qsize()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.is_running:
            return
        self._stop_flag = False
        self._thread = threading.Thread(target=self._run, name="batching-scheduler", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_flag = True
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    # ---- engine thread ----

    def _post(self, request: _Request, event):
        request.loop.call_soon_threadsafe(request.queue.put_nowait, event)

    def _finish(self, request: _Request):
        if request.finished:
            return
        request.finished = True
        with self._lock:
            self._live -= 1
        self._post(request, _END)

    def _run(self):
        try:
            while not self._stop_flag:
                self._apply_cancellations()
                self._admit(block=not self._active)
                if self._active:
                    self._step()
        except Exception:
            log.exception("scheduler thread crashed")
        finally:
            self._shutdown()

    def _apply_cancellations(self):
        with self._lock:
            cancels, self._cancels = self._cancels, []
        remove_uids = []
        for request in cancels:
            if request.finished or request.dead:
                continue
            if request.uid is None:
                # still waiting in the pending queue; skip it at admission
                request.dead = True
                self._finish(request)
            elif request.uid in self._active:
                remove_uids.append(request.uid)
                del self._active[request.uid]
                self._finish(request)
        if remove_uids:
            self.engine.remove(remove_uids)

    def _admit(self, block: bool):
        new = []
        if block:
            try:
                new.append(self._pending.get(timeout=IDLE_POLL_S))
            except queue.Empty:
                return
        while True:
            try:
                new.append(self._pending.get_nowait())
            except queue.Empty:
                break
        new = [r for r in new if not r.dead]
        if not new:
            return
        # requests with a custom stop machine are inserted separately, so plain
        # requests keep the engine's default machine
        for group, machines in (
            ([r for r in new if r.state_machine is None], None),
            ([r for r in new if r.state_machine is not None], True),
        ):
            if not group:
                continue
            kwargs = {"samplers": [r.sampler for r in group]}
            if machines:
                kwargs["state_machines"] = [r.state_machine for r in group]
            uids = self.engine.insert(
                [r.tokens for r in group],
                [r.max_tokens for r in group],
                **kwargs,
            )
            for request, uid in zip(group, uids):
                request.uid = uid
                self._active[uid] = request

    def _step(self):
        for response in self.engine.next_generated():
            request = self._active.get(response.uid)
            if request is None:
                continue
            token = response.token if response.finish_reason != "stop" else None
            self._post(request, TokenEvent(token, response.finish_reason))
            if response.finish_reason is not None:
                del self._active[response.uid]
                self._finish(request)

    def _shutdown(self):
        for request in list(self._active.values()):
            self._finish(request)
        self._active.clear()
        while True:
            try:
                self._finish(self._pending.get_nowait())
            except queue.Empty:
                break
        if hasattr(self.engine, "close"):
            self.engine.close()
