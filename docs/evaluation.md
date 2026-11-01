# Evaluation and debugging protocol

## What the public kit proves (and does not)
- `env.validate` / official `--validate`: validates tool-call schemas and availability; **does not score intent correctness**.
- `env.get_state_hash`: attests to local simulator state resulting from actions; **does not prove target state was reached**.
- `public_test.jsonl`: no labels or transcripts. It cannot directly yield ground-truth tool accuracy, task success, WER or CER.
- Do not invent benchmark scores or describe a local heuristic as an official score.

## Required reproducible checks
1. Data sanity: every input `id` unique; path exists; sample count obtained from input, not hardcoded.
2. One sample: real ToolEnv initialization, allowed tool filtering, `validate` then `call`, ordered tool-call retrieval, SDK hash retrieval.
3. Batch: exactly one output row per ID; no duplicates, missing or extra IDs; valid UTF-8 JSONL; `tool_calls` array with `{name,arguments}`; opaque `state_hash` non-empty string.
4. Run official `zaic_tools --validate` against output. Its success is **schema-only**.
5. Optional isolated replay/check: create a *separate verification session* after inference and replay recorded calls, compare the resulting hash. Never feed this verification session back into the inference agent; it is not a retry.
6. If real official scorer/leaderboard provides task score, record model, prompt, code revision, dataset version, timestamp and score. No private labels leakage.

## Error taxonomy
- Audio decode / ASR / history audio transcription
- Context extraction (time zone, relative time, profile)
- Intent or multi-intent decomposition
- Tool availability or wrong app selection
- Schema, `x-one-of`, enum, type, required fields
- `x-ref` grounding (IDs), fuzzy text matching
- Read-before-write and dependency/order
- No-op (`ok:true,done:false`), tool failure, transient model error
- Missing-info (`sys.ask_missing_info`) / unsupported request
- Unexpected extra writes / repeated irreversible side effects
- Submission ID/shape/count/hash consistency

## Experiment record
Per experiment capture: `{git_revision, dataset_version, split, sample_ids, model, asr_model, settings, max_steps, prompt_version, tool_validity_rate, noop_rate, failures_by_category, runtime, total_tokens_if_known, official_score_if_any}`.

Keep traces local. Summarize aggregates by default; do not commit user profile details or raw audio/transcripts to source control. Code should work with a fake deterministic model and no network for unit tests.
