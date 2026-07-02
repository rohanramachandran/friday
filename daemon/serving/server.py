"""FastAPI serving layer: POST /generate with SSE streaming over the batching scheduler.

Run with scripts/serve.sh, or directly: python3 -m serving.server (from daemon/).
"""
import json
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .scheduler import Scheduler, SchedulerFull

log = logging.getLogger("serving.server")

HOST = "127.0.0.1"
PORT = 8080


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(default=256, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    stream: bool = True
    template: bool = True  # wrap the prompt with the model's chat template
    ignore_eos: bool = False  # generate exactly max_tokens; for benchmarking


class IncrementalDecoder:
    """Decode token ids to text incrementally, holding back partial UTF-8."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._ids: List[int] = []
        self._emitted = 0

    def add(self, token: int) -> str:
        self._ids.append(token)
        text = self._tokenizer.decode(self._ids)
        if text.endswith("�"):
            return ""
        delta = text[self._emitted:]
        self._emitted = len(text)
        return delta


def create_app(scheduler: Scheduler, tokenizer, sampler_factory=None,
               no_stop_machine_factory=None) -> FastAPI:
    if sampler_factory is None:
        from mlx_lm.sample_utils import make_sampler

        def sampler_factory(temperature, top_p):
            return make_sampler(temp=temperature, top_p=top_p)

    if no_stop_machine_factory is None:
        def no_stop_machine_factory():
            from mlx_lm.generate import SequenceStateMachine
            return SequenceStateMachine()

    app = FastAPI(title="FRIDAY serving", docs_url=None, redoc_url=None)

    def encode_prompt(req: GenerateRequest) -> List[int]:
        if req.template and getattr(tokenizer, "has_chat_template", False):
            messages = [{"role": "user", "content": req.prompt}]
            try:
                return tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                return tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return tokenizer.encode(req.prompt)

    @app.post("/generate")
    async def generate(req: GenerateRequest):
        prompt_tokens = encode_prompt(req)
        sampler = sampler_factory(temperature=req.temperature, top_p=req.top_p)
        state_machine = no_stop_machine_factory() if req.ignore_eos else None
        try:
            handle = await scheduler.submit(
                prompt_tokens, max_tokens=req.max_tokens, sampler=sampler,
                state_machine=state_machine)
        except SchedulerFull:
            raise HTTPException(status_code=503, detail="server busy")

        decoder = IncrementalDecoder(tokenizer)

        if not req.stream:
            parts = []
            finish: Optional[str] = None
            completion_tokens = 0
            try:
                async for event in handle:
                    if event.token is not None:
                        completion_tokens += 1
                        parts.append(decoder.add(event.token))
                    if event.finish_reason is not None:
                        finish = event.finish_reason
            finally:
                await handle.cancel()
            return {
                "text": "".join(parts),
                "finish_reason": finish,
                "prompt_tokens": len(prompt_tokens),
                "completion_tokens": completion_tokens,
            }

        async def sse():
            finish: Optional[str] = None
            completion_tokens = 0
            try:
                async for event in handle:
                    if event.token is not None:
                        completion_tokens += 1
                        delta = decoder.add(event.token)
                        if delta:
                            yield f"data: {json.dumps({'text': delta})}\n\n"
                    if event.finish_reason is not None:
                        finish = event.finish_reason
                yield "data: " + json.dumps({
                    "done": True,
                    "finish_reason": finish,
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": completion_tokens,
                }) + "\n\n"
            finally:
                # no-op if the request completed; frees the slot on client disconnect
                await handle.cancel()

        return StreamingResponse(sse(), media_type="text/event-stream")

    @app.get("/health")
    async def health():
        body = {
            "status": "ok",
            "active": scheduler.active_count,
            "pending": scheduler.pending_count,
        }
        try:
            import mlx.core as mx
            body["peak_memory_gb"] = round(mx.get_peak_memory() / 1e9, 3)
        except ImportError:
            pass
        return body

    return app


def main():
    import uvicorn
    from mlx_lm import load
    from mlx_lm.generate import BatchGenerator

    from orchestrator.brain import MODEL_ID

    logging.basicConfig(level=logging.INFO)
    log.info("loading %s ...", MODEL_ID)
    model, tokenizer = load(MODEL_ID)
    engine = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        completion_batch_size=8,
        prefill_batch_size=4,
    )
    scheduler = Scheduler(engine, max_pending=32)
    scheduler.start()
    app = create_app(scheduler, tokenizer)
    log.info("FRIDAY serving on http://%s:%d", HOST, PORT)
    try:
        uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
    finally:
        scheduler.stop()


if __name__ == "__main__":
    main()
