# Benchmark results

Model: Qwen3-14B 4-bit. Hardware: Apple M5, 25.8 GB unified memory, macOS 26.5.1.
Workload: greedy decoding, EOS ignored, 128 tokens per request. Values are medians across runs (friday: 3, llamacpp: 3).

## Throughput

| Engine | 1 stream tok/s | 4 streams tok/s | 8 streams tok/s | Peak memory GB |
|---|---|---|---|---|
| friday | 14.2 | 39.7 | 42.2 | 8.78 |
| llamacpp | 13.2 | 30.4 | 32.6 | 11.69 |

Peak memory accounting differs by runtime: FRIDAY reports the MLX peak GPU allocation (Metal buffers are invisible to process RSS); llama.cpp reports peak process RSS, which includes its mmap'd weights. Both cover weights plus KV cache and are the best available number for each engine.

## Latency (seconds)

| Engine | Concurrency | TTFT p50 | e2e p50 | e2e p95 | e2e p99 |
|---|---|---|---|---|---|
| friday | 1 | 0.33 | 8.89 | 9.41 | 9.51 |
| friday | 4 | 0.57 | 12.92 | 13.07 | 13.07 |
| friday | 8 | 0.87 | 24.11 | 24.89 | 24.89 |
| llamacpp | 1 | 0.27 | 9.55 | 10.32 | 10.50 |
| llamacpp | 4 | 0.70 | 14.42 | 24.42 | 24.42 |
| llamacpp | 8 | 1.37 | 31.36 | 32.52 | 32.53 |

Raw per-request records: [results/](results/). Sample sizes per level are recorded in each file.
