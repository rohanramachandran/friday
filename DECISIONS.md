# Decisions

## 2026-07-19

- Published as a fresh repository. The private prototype history contained logs and machine-specific paths, so it was not migrated.
- The OCR helper ships as Swift source compiled during setup rather than a committed binary, so everything in the repo is auditable.
- README and setup now describe exactly the models the daemon loads (Qwen3-14B-4bit brain, Qwen3-VL-4B vision fallback). An earlier draft advertised a 35B MoE that the code no longer used.
- Screen understanding is OCR-first: Apple Vision plus window metadata covers most screen questions at near-zero memory cost, and the VLM is loaded only for visual queries, then freed. This keeps the resident footprint to the brain plus audio models.
- Token streaming is segmented at sentence boundaries (with a comma fallback for long clauses) and each sentence is synthesized immediately, so speech starts before generation finishes.
- Memory is two-tier: verbatim working context compacted at a token threshold into a summary, with compacted turns embedded for retrieval. Compaction and retrieval are covered by unit tests with a faked embedder.
- CI runs the pure-logic suite (parsing, segmentation, compaction, sandbox) on macOS runners. Inference paths are verified manually on real hardware.
