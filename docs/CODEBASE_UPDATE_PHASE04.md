# LifeMate technical update — post-Phase 04

Snapshot date: 2026-11-01. This is a read-only assessment of source, tests, documentation, and existing execution records. No tests, inference, benchmark, API request, audio, raw environment state, training data, or checkpoint was accessed for this update.

## 1. Implementation status

Phases 00–04 are marked complete in `PLANS.md`. Phase 04 adds deterministic, resumable multi-sample inference, diagnostic aggregation, exact coverage checks, and atomic submission generation. Phase 05 optimization and private packaging have not started.

| Component | Status | Evidence | Limitations |
|---|---|---|---|
| Phases 00–03 runtime | Ready and verified | Prior 37-test result, real SDK tests, real Qwen 1.7B CUDA smoke | No live production LLM evidence; native-HF/CPU ASR unverified |
| Batch orchestration | Ready and verified | `lifemate/batch.py:BatchRunner`; batch tests; 634-row offline run | Full run used debug text and scripted decisions |
| Resume/checkpointing | Ready and verified | `BatchRunner._prepare/_checkpoint`; interruption/resume test | Manifest does not fingerprint catalog/SDK contents |
| Submission construction | Ready and verified | `validate_submission_records`; structural and official validation passed | Schema validity is not intent accuracy |
| OpenAI integration | Implemented but unverified | Request-shape/retry/rate-limit unit tests | No recorded successful real API request |
| Full audio + real LLM batch | Implemented but unverified | Audio batch path and shared-transcriber test | Never executed end-to-end at batch scale |
| Official accuracy evaluation | Not ready | `official_accuracy_score` is deliberately `null` | No labels or local official scorer |

There is no known Phase 04 acceptance blocker for batch plumbing. Missing work is quality evaluation, provider/competition-rule confirmation, optimization, and packaging.

## 2. Changes since the Phase 03 snapshot

New files are `lifemate/batch.py` (batch lifecycle, checkpoints, validation, metrics), `tests/test_batch.py` (six batch tests), and `scripts/phase04_smoke.py` (offline scripted real-SDK smoke). Phase 04 also updated `lifemate/model.py`, `agent.py`, `runtime.py`, `cli.py`, and `__init__.py`; extended `tests/test_model.py` and `test_cli.py`; and updated planning/build/architecture records. No functionality was removed.

`model.py` now carries typed provider errors, retry metadata, bounded exponential backoff, optional request-rate limiting, model latency, and selectable `max_tokens`/`max_completion_tokens`. `runtime.py:RunResult` now exposes model latency. The CLI gained the `batch` subcommand and changed live defaults to `https://api.openai.com/v1` and `gpt-4.1-mini`. Remote endpoints require `--allow-external-data`; the official OpenAI host additionally requires the selected API-key environment variable.

## 3. Current end-to-end architecture

```text
public_test JSONL/catalog
  -> one shared Qwen3ASRTranscriber (+ optional cache)
  -> AudioInferencePipeline: final audio + ordered history reconstruction
  -> fresh LlmDecisionSource conversation per sample
  -> Runtime: one ToolEnv -> validate -> sequential call -> observation
  -> ToolEnv.get_tool_calls() + get_state_hash()
  -> atomic progress/failures/metrics -> validated submission JSONL
```

`lifemate/cli.py:batch_main` constructs one `OpenAICompatibleClient`, one ASR transcriber, and one `Runtime`, then passes a decision-source factory to `BatchRunner`. `BatchRunner._process_sample` creates a fresh `LlmDecisionSource`, so conversations do not leak. `AudioInferencePipeline` reuses the supplied transcriber; `Qwen3ASRTranscriber.initialize` loads its backend once. With multiple workers, `_LockedTranscriber` serializes access to that one model.

`Runtime.run` creates exactly one official session at `runtime.py:145–147`. It never resets it, and submission fields come only from `session.get_tool_calls()` and `session.get_state_hash()` at lines 249–254. `executor.map` may execute concurrently but yields results in input order; checkpoints and the final file are rebuilt in canonical sample order. Exceptions and unsafe terminal reasons become failure records, not fabricated submission rows.

## 4. OpenAI-compatible LLM integration

`model.py:OpenAICompatibleClient` uses the standard-library `urllib` transport and POSTs to `<base-url>/chat/completions`. Defaults are OpenAI, `gpt-4.1-mini`, 60-second timeout, temperature `0.0`, and 1,024 output tokens. Model, URL, timeout, temperature (including omission via `none`), token-field name, retry delays/count, and requests per minute are configurable.

If a key is present, authentication is `Authorization: Bearer <key>`; custom endpoints can operate without a key. Requests use Chat Completions messages and function tools, `tool_choice=auto`, and `parallel_tool_calls=false`. Official OpenAI automatically receives `max_completion_tokens`; other endpoints receive `max_tokens` unless overridden.

HTTP 429, 408, 409, 500, 502, 503, 504, transport errors, and timeouts are transient. Defaults allow three retries with 1/2/4-second bounded exponential delays (maximum 8 seconds); numeric `Retry-After` is honored up to that cap. HTTP 400-class non-transient errors are not retried. Provider failures become safe agent termination/failure records. One client instance shares rate-limit state across samples. However, default `urllib.request.urlopen` has no explicit persistent connection pool, so TCP connection reuse is not guaranteed.

Only injected-transport unit tests verify this behavior. Existing records explicitly say no real OpenAI request succeeded or was attempted.

## 5. Batch execution

`BatchRunner.run` streams all public rows into input order, applies an optional prefix `--limit`, rejects duplicate input IDs, and defaults to one worker. Higher `--workers` values use threads; ToolEnv/model work may overlap while shared ASR remains serialized.

Every completed sample atomically rewrites `batch-progress.jsonl`, `failures.jsonl`, and `metrics.json`. `batch-manifest.json` binds resume to dataset path/split/count, selected-ID digest, input mode, and semantic run identity. Resume validates checkpoint row shape/index, skips completed IDs, and retries failed/incomplete IDs. Existing state without `--resume`, or a changed manifest, fails closed.

The requested final output is absent whenever any selected sample fails. When complete, a hidden candidate is checked for exact fields, unique IDs, coverage, order, call shapes, and non-empty hashes; the official validator then runs in a separate process before `os.replace` promotes it. Atomic writes make partial-file corruption unlikely. Main residual risks are disk exhaustion, a validator crash leaving only checkpoints, and the absence of dataset/catalog/SDK content hashes in the resume identity. Checkpoint rewriting is O(n²) total I/O over a run. Caching avoids repeated ASR work, but full-scale GPU memory and throughput have not been measured.

## 6. Evaluation and diagnostics

`metrics.json` records dataset/selected/completed/failed counts; samples with calls; invalid model proposals; schema, SDK/tool, uncertain-execution and no-op counts; clarification and unsupported counts; termination-reason histogram; average/p50/p90/p95/p99 end-to-end sample latency; ASR total/average time, turns and cache hits; LLM total/average time and request count; input/output/total tokens; readiness; and official schema-validator text. API cost estimation is not implemented.

`batch-progress.jsonl` stores per-success submission data, termination, timing, ASR/cache counts, model requests/tokens, and step statuses. `failures.jsonl` stores sample/index, error type/message, termination, diagnostics, and calls sent—but no fabricated hash. `batch-manifest.json` stores resume identity. Final `submission.jsonl` contains only `id`, `tool_calls`, and `state_hash`.

Answers: (1) yes, the code generated 634 rows offline; real audio/LLM quality is unverified. (2) yes, structural and official schema validation are local. (3) no official LifeMate accuracy can be calculated locally. (4) based on available materials, an organizer scorer/leaderboard submission is required for an official score. (5) no metric is labeled accuracy: `diagnostic_only=true` and `official_accuracy_score=null`; `ready_for_submission` means coverage/schema readiness only.

## 7. Repository-supported PowerShell commands

```powershell
# One OpenAI audio sample
$env:OPENAI_API_KEY = '<set locally>'
.\.venv-asr\Scripts\python.exe -m lifemate --input-mode audio --mode model `
  --dataset-root '.\data\train+public_test' --sample-id public_0006 `
  --model gpt-4.1-mini --allow-external-data --submission-out '.\outputs\single-openai.jsonl'

# Five samples; remove --limit for all 634
.\.venv-asr\Scripts\python.exe -m lifemate batch `
  --dataset-root '.\data\train+public_test' --input-mode audio --limit 5 `
  --model gpt-4.1-mini --workers 1 --allow-external-data `
  --output '.\outputs\openai-five\submission.jsonl'

# Full run
.\.venv-asr\Scripts\python.exe -m lifemate batch `
  --dataset-root '.\data\train+public_test' --input-mode audio `
  --model gpt-4.1-mini --workers 1 --allow-external-data `
  --output '.\outputs\openai-full\submission.jsonl'

# Resume the full run with identical semantic settings
.\.venv-asr\Scripts\python.exe -m lifemate batch `
  --dataset-root '.\data\train+public_test' --input-mode audio `
  --model gpt-4.1-mini --workers 1 --allow-external-data --resume `
  --output '.\outputs\openai-full\submission.jsonl'

# Custom local OpenAI-compatible endpoint
$env:LOCAL_LLM_API_KEY = '<optional local key>'
.\.venv-asr\Scripts\python.exe -m lifemate batch `
  --dataset-root '.\data\train+public_test' --input-mode audio --limit 5 `
  --base-url 'http://127.0.0.1:8000/v1' --model 'local-model' `
  --api-key-env LOCAL_LLM_API_KEY --output '.\outputs\local-five\submission.jsonl'

# Tests (supported; not rerun for this report)
$env:UV_CACHE_DIR='.uv-cache'; uv run --frozen pytest -q
uv run --frozen ruff check lifemate tests scripts

# Inspect diagnostics
Get-Content '.\outputs\openai-full\metrics.json'
Get-Content '.\outputs\openai-full\failures.jsonl'

# Structural validation
uv run --frozen python '.\scripts\verify_submission.py' `
  --dataset-root '.\data\train+public_test' --submission '.\outputs\openai-full\submission.jsonl'

# Manual official schema validation (batch already does this automatically)
$python = (Resolve-Path '.\.venv-asr\Scripts\python.exe').Path
$root = (Resolve-Path '.\data\train+public_test').Path
$submission = (Resolve-Path '.\outputs\openai-full\submission.jsonl').Path
Push-Location '.\data\train+public_test\public_test'
& $python -m zaic_tools --split public_test --root $root --validate $submission
Pop-Location
```

Batch completion already invokes official `zaic_tools --validate`; `--skip-official-validation` exists only for controlled local/debug use. A remote custom endpoint also requires `--allow-external-data`.

## 8. Test and execution evidence

The last recorded suite is 48 passed in 0.42–0.43 seconds, Ruff clean, compileall successful. Six batch tests cover ordering under concurrency, per-sample sessions/conversations, resume, shared ASR, SDK-derived calls/hashes, malformed coverage, and terminal failure exclusion. Five marked integration tests exercise the real public SDK. Provider tests use fake transports; no live LLM/API test exists.

Recorded ASR evidence remains the Phase 03 Qwen 1.7B CUDA/BF16 smoke: about 13.1 seconds initialization, 4,480.8 MiB peak allocated VRAM, and successful final/history transcription. The Phase 04 three-row smoke and resume passed official validation. The full offline run processed 634/634 in a recorded 13.6-second command, with zero failures, zero calls, 634 scripted `unsupported` terminations, zero ASR turns/tokens, 14.66 ms mean sample latency, 19.18/21.02/22.92 ms p90/p95/p99, and `634 lines checked, 0 schema errors`. It is plumbing evidence, not a useful submission or score.

## 9. Known issues and opportunities

| Classification | Issue | Recommendation |
|---|---|---|
| Confirmed missing | No real LLM/API success evidence | Run an authorized one-sample smoke before any batch |
| Confirmed missing | No semantic scorer or official score | Use leaderboard feedback; never optimize against hashes |
| Confirmed missing | No API cost estimator | Add provider-price configuration and estimated/actual spend reports |
| Potential risk | Full history/observations lack token budgeting | Add measured truncation/summarization preserving IDs and dependencies |
| Potential risk | Resume does not fingerprint catalog/SDK/dataset contents | Add non-sensitive version/content digests |
| Potential risk | Official-validator failure may leave only pre-final metrics | Persist a distinct validation-failure status and message |
| Potential risk | ASR/history is all-or-nothing per sample | Add explicit recoverable policy without silently dropping history |
| Optimization | Atomic progress rewrites accumulate O(n²) I/O | Use append plus periodic compact snapshots or a safe journal |
| Optimization | ASR is serialized with multiple workers | Benchmark a dedicated ASR producer/batching strategy within VRAM |
| Optimization | No explicit HTTP connection pool | Adopt a tested persistent transport while retaining injection and retries |

## 10. Final readiness

| Target | Status | Assessment |
|---|---|---|
| Single-sample OpenAI smoke | Implemented but unverified | CLI and safeguards exist; no real request evidence |
| Small batch evaluation | Partially ready | Plumbing verified; first real model/ASR run still required |
| Full public-test inference | Implemented but unverified | Complete offline plumbing; cost, throughput, quality unmeasured |
| Public leaderboard submission | Not ready | Must produce a real-model result and obtain explicit submission authorization |
| Phase 05 optimization | Partially ready | Diagnostics exist, but no semantic baseline/score yet |
| Private-test Docker packaging | Not ready | Private protocol, network/model rules, and packaging are unverified |

