# Benchmarks

FRIDAY's serving layer against a llama.cpp server baseline, same model class
(Qwen3-14B at 4-bit), same fixed-length greedy workload.

- [bench.py](bench.py): the harness. Streams requests at concurrency 1/4/8,
  records per-request TTFT, end-to-end latency, and token counts, and samples
  the serving process's RSS. EOS is ignored so every request generates exactly
  the same number of tokens, which keeps latency comparable across engines.
- [report.py](report.py): medians across runs, rendered to [RESULTS.md](RESULTS.md).
- [results/](results/): raw per-request JSON, one file per run, committed as measured.

## Reproducing

```bash
# FRIDAY (terminal 1, then run the harness in terminal 2)
./scripts/serve.sh
python benchmarks/bench.py --engine friday --base http://127.0.0.1:8080 --run-id run1

# llama.cpp baseline
llama-server -m <Qwen3-14B-Q4_K_M.gguf> --port 8081 -ngl 99 -c 16384 --parallel 8
python benchmarks/bench.py --engine llamacpp --base http://127.0.0.1:8081 --run-id run1

python benchmarks/report.py
```

## Caveats

- The quantization schemes differ: MLX 4-bit affine grouped quantization vs
  GGUF Q4_K_M. Both are 4-bit; bit-identical weights across the two runtimes
  are not possible.
- Peak RSS is process-resident memory sampled at 4 Hz; for FRIDAY the
  MLX-reported peak GPU allocation is also recorded in the raw JSON.
- One machine, no thermal controls beyond letting each engine warm up first.
  Sample sizes per level are in the raw files; p99 at n=32 is indicative only.
