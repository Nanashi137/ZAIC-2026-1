# Minimal architecture — verified Phase 04.5 hardened batch agent

```text
public_test JSONL + catalog + public audio
          │
          ├── BatchRunner: ordered selection / resume / workers
          │       └── fresh LlmDecisionSource + ToolEnv per sample
          │
          ├── text debug override ─────────────────────────────┐
          │                                                    │
          └── AudioInferencePipeline                           │
                    ├── Qwen3ASRTranscriber (one loaded model) │
                    ├── optional content-addressed cache       │
                    └── ordered reconstructed history ─────────┤
                                                               ▼
DatasetReader ── sample permission filter ── ModelToolRegistry
          │                                  │
          │                         permitted dynamic schemas only
          ▼                                  ▼
ContextBuilder ───────────────► LlmDecisionSource ◄── ModelClient
 public context/history             │                   │
 final transcript                   │             scripted (tests)
 no raw state/audio paths            │             OpenAI-compatible HTTP
                                     ▼
 Runtime: proposal → permission → env.validate → env.call
                                     │
                          actual observation returned to model
                                     │
                                     ▼
                   one ToolEnv for the complete inference
                                     │
                                     ▼
                    get_tool_calls() + get_state_hash()
```

## Implemented boundaries

| Module | Responsibility | Competition boundary |
|---|---|---|
| `lifemate/data.py` | Stream public input, parse samples, load/restrict catalog | Refuses `train`; never opens audio or environment state |
| `lifemate/sdk.py` | Load split-local official SDK and create one session | Runtime protocol deliberately omits `get_state()` |
| `lifemate/model.py` | Model request/response types, inference settings, HTTP client, deterministic fake | Standard library; tests have no network or paid API dependency |
| `lifemate/agent.py` | Context builder, safe tool-name map, dynamic schemas, model decision source | Receives only public input, permitted tools, and actual runtime observations |
| `lifemate/runtime.py` | Bounded proposal/validation/execution loop and official output capture | One session; invalid proposals do not execute; uncertain writes terminate |
| `lifemate/asr.py` | Qwen3-ASR settings, lazy reusable model, path validation, structured results, optional cache | Local inference only; no alternate ASR, hosted service, aligner, or raw-state access |
| `lifemate/audio_pipeline.py` | Final/history transcription and conversation reconstruction | History calls/results are context only and never dispatched |
| `lifemate/batch.py` | Ordered workers, checkpoints, resume, metrics/failures, final validation/export | Never invents a failed row/hash; final file requires exact coverage |
| `lifemate/cli.py` | Explicit-action or model mode for one text-debug or audio sample | Remote model endpoints require explicit `--allow-external-data`; ASR stays local |
| `scripts/phase02_smoke.py` | Offline scripted-model smoke using the real SDK | Plumbing/schema check only; never presented as model quality |
| `scripts/phase03_smoke.py` | Cached real-ASR → scripted model-loop → real-SDK smoke | End-to-end plumbing check only; never presented as model quality |
| `scripts/phase04_smoke.py` | Partial/full offline batch through real SDK and official validator | Scripted unsupported decisions; coverage plumbing, not quality |
| `scripts/smoke_qwen_asr.py` | Privacy-safe model/device/timing diagnostics | Prints transcript hashes/counts, never private transcript text |

## Model context

`ContextBuilder` supplies:

- final user text override, explicitly marked as debug text rather than public-test ground truth;
- `current_time`, timezone, location, and user profile from public input;
- assistant history text/calls/results, marked as already applied;
- ordered Qwen3-ASR transcripts for earlier user audio in audio mode, or explicit placeholders in text-debug mode;
- a reversible mapping between provider-safe function names and official dotted names.

Audio paths themselves are not sent. Raw `state.json` and `env.get_state()` are never available to the builder. History
tool calls are context only and are never dispatched by the runtime.

## Dynamic tool schemas

Only the sample's installed catalog tools plus verified `sys.ask_missing_info` are exposed. Official names such as
`smart_home.get_status` are mapped to provider-safe names such as `smart_home__get_status`, then mapped back before
permission and SDK validation.

Model-facing schemas:

- retain supported types, properties, required fields, enums, ranges, formats, items, and descriptions;
- describe official `x-one-of` groups without forbidden root-level provider unions; the unchanged SDK schema remains
  authoritative during `env.validate`;
- turn `x-ref` and `x-match` into planning instructions without claiming local referential validation;
- use `additionalProperties: false` to reduce hallucinated fields;
- explicitly use provider `strict: false`, because the official schemas contain optional properties and custom
  extensions whose semantics must remain governed by `env.validate`.

The provider schema is guidance. Official SDK validation remains authoritative for every proposed action.

## Sequential model protocol

The OpenAI-compatible client uses chat-completions function calls with `tool_choice: auto` and
`parallel_tool_calls: false`. The latter guarantees at most one proposed call per model turn. Each accepted or rejected
proposal is followed by a tool/feedback message containing the local validation result and actual SDK observation.

This follows the official OpenAI function-calling workflow: execute a function call, append its output, and continue;
the same documentation notes that disabling parallel tool calls enforces zero or one call. See
<https://developers.openai.com/api/docs/guides/function-calling>.

When no tool is needed, the prompt requires one of these final objects:

```json
{"status":"done","reason":"..."}
{"status":"unsupported","reason":"..."}
```

Malformed arguments/final output consume a step and may be retried without executing. Refusal, provider failure,
invalid-output retry exhaustion, max steps, max calls, clarification, and unsupported requests have distinct
termination reasons.

## Write safety and result handling

- Locally unavailable and schema-invalid actions never reach `env.call`.
- Identical writes that already reached the SDK are blocked within the same inference.
- If `env.call` raises and the write outcome could be uncertain, the runtime stops immediately; it neither retries nor
  creates another session.
- SDK-rejected writes and executor exception results terminate the write path.
- `ok: true, done: false` remains a no-op and is returned to the model as such.
- The official trace is never reconstructed or filtered; the final row uses `get_tool_calls()` and
  `get_state_hash()` directly.

## Provider and privacy configuration

`OpenAICompatibleClient` accepts a base URL, optional API key, timeout, retry settings, optional requests-per-minute
limit, and injected transport. `InferenceSettings` controls model, optional temperature, maximum output tokens, and
the token-limit field. The CLI reads credentials from a named environment variable; it never writes credentials.

The requested live-test defaults are `https://api.openai.com/v1`, `OPENAI_API_KEY`, and `gpt-4.1-mini`. Official
OpenAI documentation confirms Chat Completions and function calling support for that model, while currently marking
the model deprecated. The official endpoint uses `max_completion_tokens`; non-OpenAI compatible endpoints default to
legacy `max_tokens`. `--token-limit-field` overrides this choice, and `--temperature none` omits sampling temperature
for model families that do not accept it. See <https://developers.openai.com/api/docs/models/gpt-4.1-mini> and
<https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create>.

Local example:

```powershell
uv run python -m lifemate `
  --mode model --base-url 'http://127.0.0.1:8000/v1' --model 'local-model' `
  --dataset-root '.\data\train+public_test' --sample-id public_0039 `
  --transcript 'debug transcript'
```

A non-local URL is rejected unless `--allow-external-data` is explicitly present. That flag is a technical guard, not
a claim that competition rules permit external processing. Confirm those rules and user approval before using it.

## Qwen3-ASR boundary

`Qwen3ASRTranscriber` has two strictly paired backends:

- `qwen-asr-transformers` for `Qwen/Qwen3-ASR-1.7B` (default) or `Qwen/Qwen3-ASR-0.6B`;
- `native-hf` only for the corresponding `Qwen/Qwen3-ASR-*-hf` checkpoint.

The default backend uses the official `qwen-asr` package, sends the resolved local audio path directly to Qwen, and
loads the model lazily once per transcriber instance. Device (`auto`, CPU, CUDA, or explicit device), dtype, language
(`Vietnamese` or detection), revision, maximum output tokens, and inference batch size are configurable. FlashAttention
2 and vLLM are not required. There is no faster-whisper fallback and no timestamp/forced-aligner path.

The optional cache key is derived from audio bytes plus checkpoint, revision, backend, language, token limit, dtype,
and preprocessing version. Cache files live under ignored local storage; failed or empty transcriptions are never
written as successes. A cache hit avoids model initialization entirely.

`AudioInferencePipeline` resolves references only under the selected public split. It transcribes final audio, then
prior user audio in original order, preserves assistant text and already-applied tool records, and calls the existing
runtime with reconstructed history. It does not instantiate `ToolEnv`; the runtime remains the sole owner of one
official session for current-turn execution.

## Windows environment

The verified native setup is an isolated Python 3.12 environment with `qwen-asr==0.0.6`,
`transformers==4.57.6`, `accelerate==1.12.0`, and `torch==2.12.1+cu130`. The locked `asr` dependency group pins the
official PyTorch CUDA 13.0 index. The ordinary Phase 02 environment remains ASR-free because imports are lazy.

```powershell
$env:UV_CACHE_DIR = '.uv-cache'
uv venv --python 3.12 .venv-asr
uv pip install --python .venv-asr\Scripts\python.exe --group asr

$env:HF_HOME = (Resolve-Path .).Path + '\cache\huggingface'
.\.venv-asr\Scripts\python.exe -m lifemate `
  --input-mode audio --mode model `
  --dataset-root '.\data\train+public_test' --sample-id public_0006 `
  --model 'local-agent-model' --base-url 'http://127.0.0.1:8000/v1'
```

CPU remains supported with `--asr-device cpu --asr-dtype float32`, subject to practical latency and memory. The native
Hugging Face backend requires Transformers 5.13.0+ and was implemented but not used in the successful smoke because the
official package backend worked on Windows. Phase 03 does not batch the split, optimize against hidden outcomes, or
claim WER, CER, task success, or an official quality score.

## Batch lifecycle and outputs

`BatchRunner` materializes only public sample metadata, then processes a selected prefix or the complete split. One
decision-source factory call prevents conversation leakage; one shared HTTP client preserves connections/rate state;
one shared transcriber preserves Qwen weights. With multiple workers, Qwen calls are lock-serialized while independent
model/ToolEnv work may overlap. Results are emitted in original input order regardless of completion order.

The output directory contains:

- `batch-manifest.json`: resume identity without API secrets;
- `batch-progress.jsonl`: atomic successful-sample checkpoint including diagnostics and SDK submission fields;
- `failures.jsonl`: exception or unsuccessful-agent termination records, never fabricated submission rows;
- `metrics.json`: diagnostic-only aggregates and official-schema-validation result;
- requested `submission.jsonl`: created atomically only after exact ID/order/shape checks and official validation.

Resume verifies the selected-ID digest and semantic settings, loads completed records, and retries only incomplete or
failed IDs. A completed ID is stored once. Transport retries apply only to safe model HTTP requests; uncertain tool
execution still terminates without retrying or resetting its `ToolEnv`.

Text batch mode is intentionally debug-only and requires a separate JSONL file with `{id, transcript}` rows. Public
audio mode remains the real pipeline. A non-local model endpoint requires `--allow-external-data`; this guard does not
assert that competition rules permit external processing.

## Supported Phase 04 commands

Single live OpenAI audio sample (paid/external; run only with authorization):

```powershell
$env:OPENAI_API_KEY = '<set outside source control>'
.\.venv-asr\Scripts\python.exe -m lifemate `
  --input-mode audio --mode model --dataset-root '.\data\train+public_test' `
  --sample-id public_0006 --base-url 'https://api.openai.com/v1' `
  --api-key-env OPENAI_API_KEY --model gpt-4.1-mini --allow-external-data `
  --submission-out '.\outputs\single-openai.jsonl'
```

Five-sample batch (use a new output directory):

```powershell
.\.venv-asr\Scripts\python.exe -m lifemate batch `
  --dataset-root '.\data\train+public_test' --split public_test --input-mode audio `
  --base-url 'https://api.openai.com/v1' --api-key-env OPENAI_API_KEY `
  --model gpt-4.1-mini --limit 5 --workers 1 --allow-external-data `
  --output '.\outputs\openai-five\submission.jsonl'
```

Omit `--limit` for all 634 samples. Resume the same command with `--resume`; semantic settings and selected sample set
must match the manifest. `--requests-per-minute`, `--timeout`, and bounded retry flags may be tuned without exposing
the API key. Development batch completion performs official schema validation.

## Phase 04.5 competition controls

`--competition-mode` separates scored inference from development workflows. It defaults each sample to a 28-second
`time.monotonic()` deadline, rejects non-local model URLs, seeds available Python/NumPy/PyTorch/CUDA runtimes with
seed 42, explicitly configures greedy ASR generation, and disables the post-batch official validator. The validator
remains available in development mode or as a separate command, but it must not run inside scored inference because
its CLI constructs another ToolEnv for each row.

The same `SampleDeadline` covers sample setup, final/history ASR, model turns, retries/rate-limit waits, tool
validation/execution, and SDK output capture. HTTP request timeout is capped to remaining time; a retry or wait is
rejected when it cannot fit. Expiration is recorded as `sample_timeout` and the batch never promotes a failed sample
to the final submission.

The official `qwen-asr==0.0.6` Transformers wrapper exposes no timeout argument and performs synchronous CUDA
generation. Cooperative checks can detect but cannot interrupt an overrun. `scripts/competition_watchdog.py` therefore
runs one scored sample in a dedicated child process and kills that process at the outer deadline. Wrapping an entire
multi-sample batch in one watchdog is not equivalent. Phase 05 packaging must wire one child/watchdog per scored
sample or an organizer-verified equivalent.

```powershell
# Development: separate validator remains enabled.
.\.venv\Scripts\python.exe -m scripts.phase04_smoke `
  --output '.\outputs\development-smoke\submission.jsonl'

# Competition hard-stop pattern for one sample. The final local LLM command is a Phase 05 dependency.
.\.venv\Scripts\python.exe -m scripts.competition_watchdog --timeout-seconds 30 -- `
  .\.venv-asr\Scripts\python.exe -m lifemate --competition-mode `
  --sample-timeout-seconds 28 --seed 42 --input-mode audio --mode model `
  --base-url 'http://127.0.0.1:8000/v1' --model '<phase05-local-model>' `
  --dataset-root '.\data\train+public_test' --sample-id public_0006
```

Setting `PYTHONHASHSEED` from inside a running process only affects child processes; the final Docker entrypoint must
also export it before Python starts. PyTorch deterministic mode can warn or fail when an operation has no deterministic
implementation. These limitations are surfaced rather than treated as guaranteed bitwise CUDA reproducibility.
