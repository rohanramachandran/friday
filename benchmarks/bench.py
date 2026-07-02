"""Benchmark harness: FRIDAY serving layer vs a llama.cpp server baseline.

Identical fixed-length workloads against either engine's HTTP API:
tokens/sec at concurrency 1/4/8, time to first token, end-to-end latency
percentiles, and peak resident memory of the serving process.

Usage:
  python benchmarks/bench.py --engine friday   --base http://127.0.0.1:8080 --run-id run1
  python benchmarks/bench.py --engine llamacpp --base http://127.0.0.1:8081 --run-id run1

Raw results land in benchmarks/results/<engine>-<run-id>.json.
Generation is greedy (temperature 0) with EOS ignored, so every request
produces exactly --max-tokens tokens and latency is comparable across engines.
"""
import argparse
import asyncio
import json
import platform
import statistics
import subprocess
import threading
import time
from pathlib import Path

import httpx

MAX_TOKENS = 128
LEVELS = [(1, 16), (4, 32), (8, 32)]  # (concurrency, num requests)
WARMUP_REQUESTS = 2

PROMPTS = [
    "The history of the transistor begins with a series of experiments at Bell Labs where",
    "A limit order book keeps track of outstanding buy and sell orders by",
    "In distributed systems, the problem of consensus arises whenever multiple machines must",
    "The Metal shading language differs from CUDA in several important ways, starting with",
    "Photosynthesis converts light energy into chemical energy through a process that",
    "The Raft algorithm decomposes consensus into leader election, log replication, and",
    "Unified memory on Apple Silicon means the CPU and GPU share a single pool of",
    "A Bloom filter is a probabilistic data structure that answers membership queries by",
    "The attention mechanism in transformers computes a weighted sum of values where",
    "Market makers profit from the bid-ask spread but face adverse selection when",
    "Speculative decoding accelerates language model inference by letting a small draft model",
    "The key insight behind quantization of neural networks is that weights can be stored",
    "Continuous batching improves serving throughput because new requests can join",
    "The difference between a process and a thread comes down to what they share:",
    "Backpropagation computes gradients efficiently by applying the chain rule",
    "An operating system scheduler must balance fairness against throughput when",
    "The CAP theorem states that a distributed data store can only guarantee two of",
    "Branch prediction matters for performance because modern CPUs pipeline",
    "In reinforcement learning, the exploration-exploitation tradeoff describes",
    "A memory allocator that avoids fragmentation typically organizes free blocks by",
    "The Fourier transform decomposes a signal into its constituent frequencies by",
    "Cache coherence protocols like MESI ensure that multiple cores see",
    "Grammar-constrained decoding forces a language model's output to conform to",
    "The kelly criterion tells a gambler how much of their bankroll to wager by",
    "Copy-on-write is an optimization where memory pages are shared until",
    "The two-phase commit protocol coordinates distributed transactions by",
    "Rotary position embeddings encode token positions in transformers by",
    "A lock-free queue achieves thread safety without mutexes by relying on",
    "The roofline model characterizes performance limits by plotting arithmetic intensity against",
    "Order flow toxicity measures the information content of trades and predicts",
    "Paged attention reduces KV cache fragmentation in LLM serving by",
    "The halting problem demonstrates a fundamental limit of computation because",
]


def percentile(values, p):
    xs = sorted(values)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class MemorySampler:
    """Samples RSS of the serving process in a background thread."""

    def __init__(self, pid):
        import psutil
        self._proc = psutil.Process(pid) if pid else None
        self.peak_rss = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                self.peak_rss = max(self.peak_rss, rss)
            except Exception:
                return
            self._stop.wait(0.25)

    def start(self):
        if self._proc:
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()


async def request_friday(client, base, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    async with client.stream("POST", base + "/generate", json={
            "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0,
            "top_p": 1.0, "template": False, "ignore_eos": True, "stream": True}) as r:
        r.raise_for_status()
        buf = ""
        async for chunk in r.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                if not block.startswith("data: "):
                    continue
                ev = json.loads(block[6:])
                if "text" in ev and ttft is None:
                    ttft = time.perf_counter() - t0
                if ev.get("done"):
                    tokens = ev["completion_tokens"]
    return {"ttft": ttft, "e2e": time.perf_counter() - t0, "tokens": tokens}


async def request_llamacpp(client, base, prompt, max_tokens):
    t0 = time.perf_counter()
    ttft = None
    tokens = 0
    async with client.stream("POST", base + "/completion", json={
            "prompt": prompt, "n_predict": max_tokens, "temperature": 0.0,
            "top_k": 1, "ignore_eos": True, "stream": True,
            "cache_prompt": False}) as r:
        r.raise_for_status()
        buf = ""
        async for chunk in r.aiter_text():
            buf += chunk
            while "\n\n" in buf:
                block, buf = buf.split("\n\n", 1)
                if not block.startswith("data: "):
                    continue
                ev = json.loads(block[6:])
                if ev.get("content") and ttft is None:
                    ttft = time.perf_counter() - t0
                if ev.get("stop"):
                    tokens = ev.get("tokens_predicted", 0)
    return {"ttft": ttft, "e2e": time.perf_counter() - t0, "tokens": tokens}


REQUEST_FNS = {"friday": request_friday, "llamacpp": request_llamacpp}


async def run_level(client, fn, base, concurrency, n, max_tokens):
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def one(i):
        prompt = PROMPTS[i % len(PROMPTS)]
        async with sem:
            results.append(await fn(client, base, prompt, max_tokens))

    t0 = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(n)))
    wall = time.perf_counter() - t0

    tokens = [r["tokens"] for r in results]
    ttfts = [r["ttft"] for r in results if r["ttft"] is not None]
    e2es = [r["e2e"] for r in results]
    decode_tps = [
        (r["tokens"] - 1) / (r["e2e"] - r["ttft"])
        for r in results if r["ttft"] is not None and r["e2e"] > r["ttft"] and r["tokens"] > 1
    ]
    return {
        "concurrency": concurrency,
        "n": n,
        "wall_s": round(wall, 3),
        "total_tokens": sum(tokens),
        "aggregate_tps": round(sum(tokens) / wall, 2),
        "decode_tps_median": round(statistics.median(decode_tps), 2) if decode_tps else None,
        "ttft_s": {"p50": percentile(ttfts, 50), "p95": percentile(ttfts, 95)},
        "e2e_s": {"p50": percentile(e2es, 50), "p95": percentile(e2es, 95), "p99": percentile(e2es, 99)},
        "requests": results,
    }


def find_pid(port):
    try:
        out = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0]) if out else None
    except Exception:
        return None


def hardware_info():
    def sysctl(key):
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_gb": round(int(sysctl("hw.memsize")) / 1e9, 1),
        "macos": platform.mac_ver()[0],
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["friday", "llamacpp"], required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--model-label", default="Qwen3-14B 4-bit")
    ap.add_argument("--engine-version", default="")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "results"))
    args = ap.parse_args()

    fn = REQUEST_FNS[args.engine]
    port = int(args.base.rsplit(":", 1)[1])
    sampler = MemorySampler(find_pid(port)).start()

    async with httpx.AsyncClient(timeout=600) as client:
        for _ in range(WARMUP_REQUESTS):
            await fn(client, args.base, PROMPTS[0], 16)

        levels = []
        for concurrency, n in LEVELS:
            level = await run_level(client, fn, args.base, concurrency, n, args.max_tokens)
            print(f"[{args.engine} {args.run_id}] c={concurrency}: "
                  f"{level['aggregate_tps']} tok/s aggregate, "
                  f"e2e p50 {level['e2e_s']['p50']:.2f}s")
            levels.append(level)

        mlx_peak = None
        if args.engine == "friday":
            health = (await client.get(args.base + "/health")).json()
            mlx_peak = health.get("peak_memory_gb")

    sampler.stop()
    out = {
        "meta": {
            "engine": args.engine,
            "engine_version": args.engine_version,
            "model": args.model_label,
            "run_id": args.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "max_tokens": args.max_tokens,
            "greedy": True,
            "ignore_eos": True,
            "hardware": hardware_info(),
            "peak_rss_gb": round(sampler.peak_rss / 1e9, 3) if sampler.peak_rss else None,
            "mlx_peak_memory_gb": mlx_peak,
        },
        "levels": levels,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.engine}-{args.run_id}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
