"""Summarize benchmark runs: median across runs per engine, markdown tables.

Usage: python benchmarks/report.py [results_dir]
Writes benchmarks/RESULTS.md and prints it.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_runs(results_dir):
    runs = defaultdict(list)
    for path in sorted(results_dir.glob("*.json")):
        data = json.loads(path.read_text())
        runs[data["meta"]["engine"]].append(data)
    return runs


def median_of(runs, getter):
    values = [getter(r) for r in runs]
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def level(run, concurrency):
    for lv in run["levels"]:
        if lv["concurrency"] == concurrency:
            return lv
    return None


def fmt(v, nd=2, suffix=""):
    return f"{v:.{nd}f}{suffix}" if v is not None else "n/a"


def main():
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
    runs = load_runs(results_dir)
    if not runs:
        print("no results found")
        return

    any_run = next(iter(runs.values()))[0]
    meta = any_run["meta"]
    n_runs = {e: len(rs) for e, rs in runs.items()}

    lines = []
    lines.append("# Benchmark results")
    lines.append("")
    lines.append(f"Model: {meta['model']}. Hardware: {meta['hardware']['chip']}, "
                 f"{meta['hardware']['memory_gb']} GB unified memory, macOS {meta['hardware']['macos']}.")
    lines.append(f"Workload: greedy decoding, EOS ignored, {meta['max_tokens']} tokens per request. "
                 f"Values are medians across runs ({', '.join(f'{e}: {n}' for e, n in n_runs.items())}).")
    lines.append("")
    lines.append("## Throughput")
    lines.append("")
    lines.append("| Engine | 1 stream tok/s | 4 streams tok/s | 8 streams tok/s | Peak memory GB |")
    lines.append("|---|---|---|---|---|")
    for engine, rs in runs.items():
        row = [engine]
        for c in (1, 4, 8):
            row.append(fmt(median_of(rs, lambda r: level(r, c)["aggregate_tps"]), 1))
        row.append(fmt(median_of(
            rs, lambda r: r["meta"].get("mlx_peak_memory_gb") or r["meta"]["peak_rss_gb"]), 2))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("Peak memory accounting differs by runtime: FRIDAY reports the MLX peak GPU "
                 "allocation (Metal buffers are invisible to process RSS); llama.cpp reports "
                 "peak process RSS, which includes its mmap'd weights. Both cover weights "
                 "plus KV cache and are the best available number for each engine.")
    lines.append("")
    lines.append("## Latency (seconds)")
    lines.append("")
    lines.append("| Engine | Concurrency | TTFT p50 | e2e p50 | e2e p95 | e2e p99 |")
    lines.append("|---|---|---|---|---|---|")
    for engine, rs in runs.items():
        for c in (1, 4, 8):
            lines.append("| " + " | ".join([
                engine, str(c),
                fmt(median_of(rs, lambda r: level(r, c)["ttft_s"]["p50"])),
                fmt(median_of(rs, lambda r: level(r, c)["e2e_s"]["p50"])),
                fmt(median_of(rs, lambda r: level(r, c)["e2e_s"]["p95"])),
                fmt(median_of(rs, lambda r: level(r, c)["e2e_s"]["p99"])),
            ]) + " |")
    lines.append("")
    lines.append("Raw per-request records: [results/](results/). "
                 "Sample sizes per level are recorded in each file.")
    lines.append("")

    text = "\n".join(lines)
    out = Path(__file__).parent / "RESULTS.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
