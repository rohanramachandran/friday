# Decisions

## 2026-07-19 (serving layer)

- The HTTP serving layer is a second entrypoint beside the voice daemon, sharing the model id but not a process. The voice path is single-user and latency-bound; coupling it to the batch scheduler bought nothing yet, so that refactor is deferred until the daemon needs concurrent generation.
- Continuous batching builds on mlx_lm's BatchGenerator rather than reimplementing batched KV cache management. Our scheduler owns what the engine does not: async admission at token boundaries, per-request streaming fan-out, cancellation on disconnect, and backpressure (503 at capacity).
- The engine is synchronous and single-threaded by design, so it lives on a dedicated thread; tokens cross into asyncio via call_soon_threadsafe onto per-request queues. No locks are held during compute.
- Serving tests run against a fake engine that mirrors the BatchGenerator contract (insert/next_generated/remove, stop-token suppression semantics), so CI exercises scheduling logic without model weights.
- Detokenization is per request via full-prefix decode with a replacement-character holdback for split UTF-8. Quadratic in output length but trivial at these sizes; revisit only if profiles say so.
- Fixed the tool-loop double answer: the forced summary phase now runs only when the first pass produced no text after tool results. Regression-tested with a scripted stream.
- Live smoke on the M5: four concurrent streams reached 2.17x the single-stream decode throughput with all streams interleaving. Rigorous numbers belong to the benchmark harness, not this log.

## 2026-07-19 (initial publish)

- Published as a fresh repository. The private prototype history contained logs and machine-specific paths, so it was not migrated.
- The OCR helper ships as Swift source compiled during setup rather than a committed binary, so everything in the repo is auditable.
- README and setup now describe exactly the models the daemon loads (Qwen3-14B-4bit brain, Qwen3-VL-4B vision fallback). An earlier draft advertised a 35B MoE that the code no longer used.
- Screen understanding is OCR-first: Apple Vision plus window metadata covers most screen questions at near-zero memory cost, and the VLM is loaded only for visual queries, then freed. This keeps the resident footprint to the brain plus audio models.
- Token streaming is segmented at sentence boundaries (with a comma fallback for long clauses) and each sentence is synthesized immediately, so speech starts before generation finishes.
- Memory is two-tier: verbatim working context compacted at a token threshold into a summary, with compacted turns embedded for retrieval. Compaction and retrieval are covered by unit tests with a faked embedder.
- CI runs the pure-logic suite (parsing, segmentation, compaction, sandbox) on macOS runners. Inference paths are verified manually on real hardware.
