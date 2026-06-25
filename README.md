# FRIDAY

![tests](https://github.com/rohanramachandran/friday/actions/workflows/ci.yml/badge.svg)

Local voice AI assistant for macOS. Everything runs on-device: speech recognition, the language model, tool execution, and speech synthesis. No cloud calls, no API keys, and it works with WiFi off.

## What it is

- **Brain**: Qwen3-14B (4-bit MLX) with a tool-use loop
- **Screen understanding**: OCR-first via Apple Vision, with a small VLM (Qwen3-VL-4B) loaded only for truly visual questions, then unloaded
- **STT**: whisper.cpp small.en (Metal-accelerated)
- **TTS**: Kokoro-82M, streamed sentence by sentence for low perceived latency
- **Tools**: screen capture, Python execution, AppleScript system control, web search, conversation memory search
- **Serving**: FastAPI `/generate` endpoint with SSE streaming, request queuing, and continuous batching, so concurrent generations share the decode loop
- **Memory**: working context with automatic compaction, plus embedding retrieval over past conversations
- **Frontends**: native SwiftUI menu bar app with global hotkeys, and a terminal client with wake-word listening

## Architecture

```mermaid
flowchart LR
    subgraph app["Swift menu bar app"]
        HK["Hotkeys + mic"] --> WSC["WebSocket client"]
        WSC --> SPK["Speaker"]
    end
    WSC <--> WSS["WebSocket server (127.0.0.1)"]
    subgraph daemon["Python daemon"]
        WSS --> STT["whisper.cpp STT"]
        STT --> BRAIN["Qwen3-14B brain (MLX)"]
        BRAIN <--> TOOLS["tools: screenshot, system, run_code, web_search, memory"]
        BRAIN --> TTS["Kokoro TTS"]
        TTS --> WSS
    end
```

The daemon streams tokens as they generate. Sentences are cut at natural boundaries and synthesized immediately, so audio starts playing while the model is still writing the rest of the answer.

Screen questions take a cheap path by default: a native Apple Vision OCR helper plus the frontmost window title, which costs roughly zero memory and about 100 ms. The VLM is loaded only when the question is actually visual (charts, photos, layout), then freed.

## Quickstart

Prerequisites: Apple Silicon Mac, Python 3.11, Xcode command line tools.

```bash
git clone https://github.com/rohanramachandran/friday
cd friday
./scripts/setup.sh        # venv, deps, model downloads (~12GB), OCR helper build
./scripts/run.sh          # starts the daemon
```

Wait for `FRIDAY ready on ws://127.0.0.1:8765`, then in another terminal:

```bash
source daemon/.venv/bin/activate
python scripts/cli.py
```

Type a question, or say "Friday" followed by a command. For the menu bar app, see [app/](app/): it is a single Swift file plus an Info.plist you drop into an Xcode macOS app target.

- **Hold Option+Space**: push-to-talk
- **Hold Option+Shift+Space**: push-to-talk with a screenshot attached
- **Menu bar icon**: toggle the overlay

## Serving API

Besides the voice daemon, the same model can be served over HTTP:

```bash
./scripts/serve.sh           # loads the model, serves on 127.0.0.1:8080
```

```bash
curl -N http://127.0.0.1:8080/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "What is a limit order book?", "max_tokens": 128}'
```

Responses stream as server-sent events; pass `"stream": false` for a single JSON body. Requests accept `temperature`, `top_p`, and `max_tokens` per call.

Under the hood a scheduler thread owns the MLX engine and admits new requests into the running batch at token boundaries (continuous batching, built on `mlx_lm`'s `BatchGenerator`). Each request streams back through its own queue, client disconnects cancel the sequence and free its batch slot, and the scheduler applies backpressure with HTTP 503 once it is at capacity.

## Tests

```bash
pip install pytest
pytest tests/
```

The suite covers tool-call parsing, streaming sentence segmentation, memory compaction, the code execution sandbox, and the serving layer (concurrent streams, continuous admission, cancellation, backpressure) against a fake engine that mirrors the batching contract. Model inference paths are exercised manually, not in CI.

## Benchmarks

The harness and raw results live in [benchmarks/](benchmarks/). Numbers published here are measured on the author's machine and committed with the raw output.

## Limitations

- macOS and Apple Silicon only, by design: the stack is built on MLX, Apple Vision, and AppleScript.
- The code execution tool runs Python in a subprocess with a timeout; it is not a hardened sandbox.
- Web search scrapes DuckDuckGo HTML and can break if the page layout changes.
- Wake-word listening transcribes a rolling two-second window with whisper tiny.en, which trades some battery for simplicity.
- A 14B brain is deliberate: it keeps headroom for the rest of the system in unified memory. Larger MLX models drop in by changing `MODEL_ID` in `daemon/orchestrator/brain.py`.
