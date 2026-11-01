# Phase 04 — Batch, submission and diagnostics

Status: COMPLETE on 2026-11-01. Phase 05 was not started.

## Deliverables
- CLI to process full public_test JSONL with per-sample isolation.
- Output JSONL with `id`, exact `env.get_tool_calls()`, and exact `env.get_state_hash()`; ensure one row per input ID.
- Deterministic output ordering, robust error reporting, atomic output strategy and clear policy for failed samples. Never output fabricated hashes.
- Run `scripts/verify_submission.py` and official `zaic_tools --validate` on output.
- Per-sample diagnostic logs: transcript, selected tools, schema errors, no-ops, execution failures, wall time, model usage (if available). Safeguard sensitive data.
- Optional offline independent replay consistency check; must never feed privileged state to agent.

## Acceptance
- Sample ID coverage equals the input; zero duplicates; legal format and SDK schema validation.
- Report pass/fail separately for structure, schema, replay consistency, and *actual official score if available*. A hash alone is not a task-success metric.
- No direct dependence on labels or training files.

## Implemented behavior

- `lifemate.batch.BatchRunner` processes selected input rows in original order, creates a fresh decision source and exactly one runtime `ToolEnv` per sample, and reuses the batch-scoped model client and Qwen transcriber.
- `workers=1` is the default. Higher worker counts preserve output order; access to one shared ASR model is serialized.
- Successful samples are atomically checkpointed in `batch-progress.jsonl`. `batch-manifest.json` binds resume to the split, selected IDs, provider/agent settings, and ASR or text-transcript identity.
- Failures are written to `failures.jsonl` without a submission row. Model errors, refusal, bad-output exhaustion, step/call exhaustion, uncertain execution, duplicate writes, and terminal SDK/tool errors reject the sample.
- `metrics.json` records diagnostic counts, termination reasons, latency percentiles, ASR/model time, cache usage, requests, and tokens. It explicitly contains no official accuracy score.
- Final `submission.jsonl` is written only when every selected ID is complete, unique, and in order. A separate process then runs the official `zaic_tools` schema validator before the candidate is promoted.
- The shared HTTP client supports request timeouts, optional requests-per-minute limiting, and bounded exponential retries for HTTP 429, timeout/network errors, and transient HTTP statuses. Tool executions are never retried.

## Verification

- Final unit/integration suite: 48 passed; Ruff and compileall passed.
- Three-sample scripted-model real-SDK smoke: 3/3 complete, resume skipped completed samples, official validation reported zero schema errors.
- Full offline scripted-model run: 634/634 rows, zero failures, deterministic coverage/order, structural checker passed, official validator reported `634 lines checked, 0 schema errors`.
- No OpenAI or other paid model endpoint was called. The full offline output contains scripted `unsupported` decisions and is not a quality submission or score.
