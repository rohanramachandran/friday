"""Model-free stand-ins for the serving tests.

FakeEngine mirrors the mlx_lm BatchGenerator contract the scheduler relies on:
insert() admits sequences and returns uids, next_generated() advances every
active sequence by one token per call, remove() drops sequences, and the
final response carries finish_reason ("length" includes its token, "stop"
does not, matching mlx_lm semantics).
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class FakeResponse:
    uid: int
    token: int
    finish_reason: Optional[str]


class FakeEngine:
    def __init__(self, eos_token: Optional[int] = None, step_delay: float = 0.0):
        self.step_delay = step_delay
        self._uid = 0
        self._active = {}  # uid -> dict(next, remaining, sampler)
        self.removed = []
        self.inserted_samplers = []
        self.closed = False
        self.eos_token = eos_token
        self.steps = 0

    def insert(self, prompts: List[List[int]], max_tokens=None, samplers=None):
        uids = []
        max_tokens = max_tokens or [128] * len(prompts)
        samplers = samplers or [None] * len(prompts)
        for prompt, m, s in zip(prompts, max_tokens, samplers):
            uid = self._uid
            self._uid += 1
            # deterministic script: tokens count up from the last prompt token
            self._active[uid] = {"next": prompt[-1] + 1, "remaining": m}
            self.inserted_samplers.append((uid, s))
            uids.append(uid)
        return uids

    def next_generated(self):
        if self.step_delay:
            import time
            time.sleep(self.step_delay)
        self.steps += 1
        responses = []
        done = []
        for uid, st in self._active.items():
            token = st["next"]
            st["next"] += 1
            st["remaining"] -= 1
            finish = None
            if self.eos_token is not None and token == self.eos_token:
                finish = "stop"
            elif st["remaining"] == 0:
                finish = "length"
            if finish:
                done.append(uid)
            responses.append(FakeResponse(uid, token, finish))
        for uid in done:
            del self._active[uid]
        return responses

    def remove(self, uids, return_prompt_caches=False):
        for uid in uids:
            self._active.pop(uid, None)
            self.removed.append(uid)
        return {}

    def close(self):
        self.closed = True


class FakeTokenizer:
    """Character-level tokenizer: token id == ord(char)."""

    eos_token_ids = [0]
    has_chat_template = False

    def encode(self, text, **kwargs):
        return [ord(c) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(chr(i) for i in ids)
