# LifeMate Technical Codebase Snapshot

Snapshot date: 2026-11-01. This report describes inspected code and existing verification records; no tests, inference, datasets, caches, generated outputs, or raw environment state were read or executed for this snapshot.

## 1. Project status

Phases 00–03 are marked complete in `PLANS.md` and have corresponding code and recorded evidence. Phase 00 documented the installed SDK contract; Phase 01 added a text-only single-sample runtime; Phase 02 added a sequential LLM/tool loop; Phase 03 added local Qwen3-ASR and audio-backed history reconstruction.

Actually implemented: public-test sample/catalog loading, sample-specific tool exposure, one official `ToolEnv` per inference, text or audio input, Qwen transcription caching, ordered history reconstruction, OpenAI-compatible chat-completions tool calling, official validation/execution, bounded continuation, and one-row SDK-derived output. The primary Qwen package backend and real SDK were smoke-tested according to `harness/build-log.md`.

Partial or unverified: the native Hugging Face `-hf` ASR backend exists in code but has no recorded real run; CPU placement is unit-tested with a fake torch object but not real-smoked; production LLM behavior is not verified because all model-loop tests/smokes use `ScriptedModelClient`.

Missing: Phase 04 full-split batch production, aggregate diagnostics, semantic evaluation, and full submission generation. `scripts/verify_submission.py` validates an already-created full file but does not create it. Phase 05 optimization/private packaging is also absent. Remaining external unknowns are exact scoring, private Docker protocol, and competition network/model policy. Next planned phase is Phase 04, not started.

## 2. Repository structure

```text
pyproject.toml / uv.lock       dependency and tool configuration
lifemate/
  data.py                     public JSONL/catalog reader and permission set
  sdk.py                      narrow official ToolEnv adapter
  model.py                    OpenAI-compatible and scripted model clients
  agent.py                    prompt, dynamic schemas, model decision state
  runtime.py                  bounded validate/call/observe controller
  asr.py                      Qwen backends, result types, local cache
  audio_pipeline.py           final/history ASR and history reconstruction
  cli.py / __main__.py        single-sample PowerShell-compatible CLI
scripts/
  phase02_smoke.py            scripted model plus real SDK smoke
  phase03_smoke.py            cached real ASR plus scripted model/SDK smoke
  smoke_qwen_asr.py           privacy-safe ASR timing/memory diagnostics
  verify_submission.py       full-file structural checker, not a scorer
tests/                        unit tests, fakes, and real-SDK integration tests
docs/                         official contract, architecture, evaluation notes
harness/build*/               phase requirements and recorded evidence
```

Datasets, virtual environments, caches, checkpoints, traces, and generated outputs are excluded from this view and ignored by `.gitignore`.

## 3. Technology stack

| Area | Actual implementation |
|---|---|
| Language | Python `>=3.11`; records show main tests on 3.13 and ASR on 3.12.12 |
| Dependency management | `uv`, `pyproject.toml`, `uv.lock` |
| ASR | `Qwen/Qwen3-ASR-0.6B` default, `qwen-asr==0.0.6`, Transformers wrapper; optional code path for `*-hf` |
| GPU stack | Locked `torch==2.12.1+cu130`; `transformers==4.57.6`, `accelerate==1.12.0` |
| LLM | No concrete model selected; configurable OpenAI-compatible Chat Completions client using `urllib` |
| Orchestration | Hand-written single-agent sequential loop; no LangGraph, planner service, or multi-agent framework |
| Structures | Frozen dataclasses, protocols, mappings, manual JSON parsing; no Pydantic dependency |
| Tool validation | Provider-facing JSON Schema subset plus authoritative official `env.validate()` |
| Execution | Installed split-local `zaic_tools.ToolEnv` |
| CLI | Standard-library `argparse`; one sample per process |
| Logging/cache | In-memory `StepResult` diagnostics and JSON to stdout/stderr; atomic local JSON transcript cache; no logging framework |
| Tests | pytest 8.4.2; Ruff 0.17.3; deterministic fakes plus marked real-SDK tests |

## 4. End-to-end inference flow

```text
public JSONL/catalog
  -> DatasetReader.get_sample/load_catalog
  -> [audio] AudioInferencePipeline -> Qwen3ASRTranscriber/CachedTranscriber
  -> reconstructed history + final transcript
  -> ContextBuilder + ModelToolRegistry
  -> LlmDecisionSource -> OpenAICompatibleClient
  -> Runtime permission check -> env.validate -> env.call
  -> actual observation -> next model turn (bounded loop)
  -> env.get_tool_calls + env.get_state_hash -> one submission row
```

1. `lifemate/data.py:DatasetReader` streams `<split>.jsonl`, parses `Sample`, and loads the catalog. `Catalog.restricted_to()` retains installed tool names plus `sys.ask_missing_info`.
2. `AudioInferencePipeline._audio_path()` resolves references beneath `dataset.split_dir`; `resolve_audio_path()` additionally requires a file and rejects root escape.
3. `Qwen3ASRTranscriber.transcribe()` hashes audio, lazy-loads one backend, transcribes the direct path, and returns `TranscriptionResult`.
4. `AudioInferencePipeline._transcribe_history()` processes prior user audio in order and preserves assistant text/tool calls/results. It never dispatches history calls.
5. `ContextBuilder.initial_messages()` creates a fixed system instruction and compact JSON user payload containing public context, reconstructed history, final transcript/source, and tool-name mapping.
6. `LlmDecisionSource.next_action()` sends current messages and permitted tool definitions. The provider chooses zero or one function because the request sets `parallel_tool_calls=false`.
7. `Runtime.run()` rejects unavailable tools, then calls the official session's `validate(name, arguments)`.
8. Valid actions reach `session.call()` sequentially. `StepResult` stores validation, observation, status, and errors.
9. `LlmDecisionSource._append_new_step_feedback()` sends the real result back as a tool message; the loop continues until a terminal result or budget/error stop.
10. `Runtime` constructs the row only from `session.get_tool_calls()` and `session.get_state_hash()`.

## 5. Agent implementation details

There is no `AgentController` class. The controller role is split between `runtime.py:Runtime` (lifecycle, budgets, validation/execution) and `agent.py:LlmDecisionSource` (model conversation and parsing). This is a single reactive agent, not a separate planner/executor architecture.

`ModelToolRegistry.from_allowed()` converts only permitted catalog entries to provider functions. Dotted names become safe names such as `sms__send`; mappings convert calls back. The schema retains common type/enum/range/format fields, translates `x-one-of` into `anyOf` required groups, and turns `x-ref`/`x-match` into descriptions. Provider `strict` is false; the official SDK remains authoritative.

The model either returns one tool call with JSON-object arguments or final JSON `{"status":"done|unsupported","reason":"..."}`. Invalid JSON, multiple calls, empty replies, or invalid final status become retryable bad proposals. Default `max_invalid_proposals=2`; a third consecutive invalid proposal terminates. Defaults are eight total steps and six calls. Bad/schema/permission proposals consume steps but not the SDK call budget.

Actual observations—including no-op and validation failures—are appended before the next model turn, enabling dependent read→write requests. History influences decisions through transcripts and assistant records marked `tool_calls_already_applied`. Missing data is handled by model selection of catalog tool `sys.ask_missing_info`; `waiting_for_user=true` terminates as `awaiting_user`. Unsupported requests rely on the model's final `unsupported` status and produce no call.

Termination reasons include model completion/unsupported, awaiting user, refusal, provider error, bad-output limit, step/call exhaustion, duplicate write, SDK rejection/tool error for writes, and uncertain execution. Identical writes already sent in the inference are blocked. An exception from `session.call()` stops immediately; there is no automatic transport/model retry or ToolEnv reset.

## 6. Qwen3-ASR implementation

Configured defaults in `Qwen3ASRSettings`: `Qwen/Qwen3-ASR-0.6B`, `qwen-asr-transformers`, device/dtype `auto`, forced `Vietnamese`, 256 output tokens, batch size 1, and preprocessing identity `qwen-direct-path-v1`. Auto placement chooses `cuda:0` then BF16 when supported, otherwise FP16; CPU defaults to FP32. Audio is passed directly to Qwen without repository-side resampling, alignment, or timestamps. The earlier Phase 03 real smoke used 1.7B; that historical evidence is not evidence for the current 0.6B default.

`_QwenPackageBackend` calls `Qwen3ASRModel.from_pretrained()` once and reuses the object for final and history files within the pipeline. Output contains stripped text, detected language when available, latency, audio SHA-256, checkpoint/backend, and cache-hit flag. Empty/malformed output and dependency/load/inference failures raise explicit ASR exceptions.

`CachedTranscriber` keys successful JSON entries by audio content plus checkpoint, configured revision, backend, language, token limit, dtype, and preprocessing version; writes are temporary-file plus `os.replace`. Failed transcriptions are not cached. Cached transcript text is stored locally in plaintext under ignored `cache/`.

Verified-run settings differ from defaults: CUDA, BF16, 128 tokens on an RTX 5060. Existing records report approximately 13.1 s initialization, 4,480.8 MiB peak allocated VRAM, 2.68–5.61 s final-audio inference, roughly 0.49–0.98 s history time, and a 3/3 repeat cache hit. Forced Vietnamese and auto detection matched by transcript hash for two clips. No reference transcripts exist, so WER/CER is not available. Native-HF and real CPU inference are **Not verified**.

## 7. Official ToolEnv integration

`OfficialSessionFactory.create()` imports `zaic_tools` only from the selected split and calls `ToolEnv(split, sample_id, root=dataset_root)`. `Runtime.run()` invokes the factory exactly once. It compares `session.allowed` to the public sample's installed tools plus `sys.ask_missing_info`, failing closed on mismatch.

Every permitted proposal is preflighted with `validate`; invalid calls never reach `call`. Returned observations are retained verbatim in `StepResult` mappings. `ok:false` is `sdk_rejected`; `done:false` is `no_op`, except an exception-prefixed message becomes `tool_error`. Writes with SDK rejection/tool error stop; reads may be reconsidered by the model. The system tool requires official catalog arguments, changes no app state, and stops when it reports `waiting_for_user`.

Final call order is never reconstructed or filtered. Rejected calls that reached the SDK remain visible through `get_tool_calls()`. Historical calls are context-only. The runtime protocol intentionally omits `get_state()`.

Correctness concerns: `Catalog.restricted_to()` silently omits a sample-allowed name absent from the catalog; catalog completeness is not separately asserted. Also, an observation missing both `ok:false` and `done:false` is classified as success according to tool kind, relying on the official SDK's normal response shape.

## 8. Configuration and execution

Important files are `pyproject.toml`, `uv.lock`, `.gitignore`, and the phase/docs records. Environment variable names used by supported workflows are `OPENAI_API_KEY` (or a name selected by `--api-key-env`), `UV_CACHE_DIR`, `HF_HOME`, and `HF_HUB_OFFLINE`. No secret values are committed.

Defaults: dataset `data/train+public_test`, split `public_test`, local endpoint `http://127.0.0.1:8000/v1`, temperature 0, LLM output 1024 tokens, timeout 60 s, ASR cache `cache/transcriptions`, and optional output path supplied by `--submission-out`.

```powershell
# Text-only model inference for one sample
uv run python -m lifemate --input-mode text --mode model `
  --dataset-root '.\data\train+public_test' --sample-id public_0039 `
  --transcript 'debug transcript' --model 'local-model'

# Audio model inference for one sample
.\.venv-asr\Scripts\python.exe -m lifemate --input-mode audio --mode model `
  --dataset-root '.\data\train+public_test' --sample-id public_0006 `
  --model 'local-model' --asr-device cuda --asr-dtype bfloat16

# Deterministic one-sample execution and one-row output
uv run python -m lifemate --sample-id public_0039 --transcript 'debug' `
  --tool alarm.list --arguments-json '{}' --submission-out '.\outputs\sample.jsonl'

# Tests
$env:UV_CACHE_DIR = '.uv-cache'; uv run --frozen pytest -q

# Structural check of an already-created full submission
uv run python .\scripts\verify_submission.py `
  --dataset-root '.\data\train+public_test' --submission '.\outputs\submission.jsonl'

# Official schema validation
Push-Location '.\data\train+public_test\public_test'
python -m zaic_tools --split public_test --root .. --validate '..\..\..\outputs\sample.jsonl'
Pop-Location
```

No command currently generates a full-split `submission.jsonl`; only single-sample rows and smoke outputs are supported.

## 9. Tests and validation

The 37-test suite covers dataset restriction, provider request shape, dynamic tools, schema/permission rejection, observation-derived read→write, clarification, unsupported requests, bad model output/refusal/error, no-op, duplicate/uncertain writes, ASR initialization/reuse/path/failure/cache/placement, history order/preservation/no replay, text regression, and no-history audio.

`tests/test_sdk_integration.py` uses the real public SDK for a read, rejected-call recording, `x-one-of`, system tool, no-op/state hash, scripted model loop, and fake-ASR history pipeline. Real Qwen inference is a separate smoke, not a pytest test. No live production LLM has been called. Existing records dated 2026-11-01 report `37 passed`, Ruff clean, real 1.7B Windows inference, scripted end-to-end `model_completed`, and official validation of one row with zero schema errors. These are recorded results, not rerun for this report.

## 10. Known issues and engineering risks

Confirmed gaps: (1) no batch producer; (2) no semantic scorer/ground truth or full coverage validation result; (3) native-HF and real CPU ASR are unverified; (4) `failure_count` is always zero on returned audio results because any transcription failure raises and aborts.

Potential risks: (5) `LlmDecisionSource` is stateful and must not be reused across samples in future batch code; (6) full observations/history are sent without truncation or context-token budgeting; (7) cache revision defaults to `None`, so a mutable upstream checkpoint can reuse stale entries, and entries contain plaintext transcripts; (8) primary model loading per CLI process is expensive until a batch runner reuses one transcriber across samples; (9) provider schemas are a curated subset and nested object schemas may lose guidance, although SDK validation protects execution; (10) ASR/history is all-or-nothing—one failed historical clip prevents current-turn ToolEnv inference. No confirmed failing correctness bug is recorded in the current test/smoke set.

## 11. Current readiness assessment

| Component | Status | Evidence | Main limitation |
|---|---|---|---|
| Sample loader | Implemented and tested | `DatasetReader`; unit and SDK tests | Linear scan per `get_sample`; public CLI split only |
| ASR | Implemented and tested | Primary 1.7B real smoke; ASR unit tests | Native-HF/CPU unverified; startup cost |
| History | Implemented and tested | Ordering/preservation/replay tests; SDK smoke | One bad clip aborts sample |
| Agent loop | Implemented and tested | Scripted multi-step/read→write tests | No live-LLM quality evidence |
| Tool validation | Implemented and tested | Permission, schema, `x-one-of` tests | Provider schema is guidance subset |
| Tool execution | Implemented and tested | Real ToolEnv tests; one-session path | Depends on normal SDK observation shape |
| Submission | Partial | SDK-derived one-row output; official one-row validation | No full-split producer |
| Evaluation | Partial | Structural checker and recorded smoke metrics | No labels, scorer, WER/CER, or task score |
