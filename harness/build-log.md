# Implementation log (Codex updates after each phase)

| Date | Phase | Files changed | Commands run and evidence | Known limitations | Next action |
|---|---|---|---|---|---|
| 2026-10-30 | 00 | `docs/official-contract.md`, `docs/architecture.md`, `PLANS.md`, this log | `uv run python .\scripts\inspect_kit.py --root '.\data\train+public_test'` → 634/634 samples, 36 apps, 122 tools, no missing audio/environment/duplicate IDs; official `python -m zaic_tools ... --apps` → 122/122 implemented, 0 missing | Exact scoring weights, private Docker, and network/model policy remain unknown | Phase 01 runtime |
| 2026-10-30 | 01 | `.gitignore`, `pyproject.toml`, `uv.lock`, `lifemate/`, `tests/`, two formatting-only script edits, docs/plan/log | `uv run pytest -q` → 12 passed; `uv run ruff check lifemate tests scripts` → all checks passed; real CLI `public_0039` read produced SDK call/hash; official validator → `1 lines checked, 0 schema errors` | Deterministic explicit actions only; no LLM, ASR, batch submission, semantic score, or ground truth | Phase 02 model-backed decision loop |
| 2026-10-30 | 02 | `lifemate/model.py`, `lifemate/agent.py`, runtime/CLI/tests, `scripts/phase02_smoke.py`, architecture/plan/build docs | `uv run pytest -q` → 25 passed; `uv run ruff check lifemate tests scripts` → all checks passed; offline scripted-model real-SDK smoke → `model_completed`; official validator → `1 lines checked, 0 schema errors` | No live external provider or quality evaluation; ASR/history transcription, scoring details, network policy, and private Docker remain unknown | Phase 03 ASR + history audio |
| 2026-11-01 | 03 | `lifemate/asr.py`, `lifemate/audio_pipeline.py`, agent/runtime/CLI, ASR/history tests, smoke script, dependencies, architecture/plan/build docs | 37 tests passed; Ruff clean; real `Qwen/Qwen3-ASR-1.7B` CUDA/BF16 run transcribed final + 2 history clips; official validator: `1 lines checked, 0 schema errors`; repeat cache run: 3/3 hits | No reference transcripts or official scorer; native-HF and CPU paths configured but not real-smoked; exact competition network/private-container rules remain unknown | Stop; Phase 04 only on explicit request |
| 2026-11-01 | 04 | `lifemate/batch.py`, provider retry/rate-limit/latency support, CLI subcommand, batch/model tests, full offline smoke, architecture/plan/build docs | 48 tests passed; Ruff/compileall clean; 634-row scripted-model run passed structural coverage and official validation (`634 lines checked, 0 schema errors`) | No paid/live LLM run or semantic score; offline rows intentionally use scripted unsupported decisions; private execution/network rules remain unknown | Stop; Phase 05 only on explicit request |
| 2026-11-01 | 04.5 | compliance deadline/seed controls, ASR/model/runtime/batch/CLI integration, watchdog, environment-access hardening, tests and docs | 60 tests passed; Ruff/compileall clean; real-SDK competition single sample passed; development 3-row validator smoke passed; Windows watchdog hard-stop verified | `qwen-asr` synchronous CUDA cannot be cancelled in-process; final offline LLM and watchdog wiring are Phase 05 dependencies | Phase 05 local inference and packaging |
| 2026-11-03 | 04.6 | finish/duplicate/read-loop controls and fail-soft rows; native/constrained-JSON mode plumbing; labeled oracle/ASR evaluation and supplied challenge formula; isolated SDK full-state gold replay; action-specific JSON schemas; controller/scoring diagnostics; deep JSON-safe export; deadline propagation; sanitized root/per-mode run configs; run provenance and cache-aware latency reporting; evaluation/contract/plan docs | 100-sample user A/B artifacts reviewed; full suite: 120 passed; Ruff clean; compileall and `git diff --check` clean. Native estimate: oracle 35.3960, ASR 12.9359; constrained JSON: oracle 22.9167, ASR 10.7333. | `challenge_score_estimate` only: official fuzzy/partial/order matcher unavailable. Saved ASR artifacts lack checkpoint provenance; native incurred 228.16 s ASR while constrained reused 333/333 cached turns, so saved wall-latency delta is not a fair mode comparison. Scores are partial train evidence only, not a public-score predictor. | Keep native mode; evaluate generic ASR/planning improvements on multiple fixed, stratified train folds before another public submission. |
| 2026-11-03 | 05 (ASR benchmark) | Refactored `lifemate/asr.py` into an import-compatible package; added PhoWhisper, Transformers Whisper, faster-whisper and ChunkFormer adapters; standalone fixed-subset benchmark, mandatory exact parameter count/provenance, tqdm, strict duration preflight, fsync checkpoint/resume, common WER/CER normalization, metrics/report, nine-run script, locked dependencies, tests and docs | Full suite: 126 passed; Ruff clean; compileall and `git diff --check` clean. Real dependency/API smoke: torch 2.12.1+cu126, torchaudio 2.11.0+cu126, chunkformer 1.2.2, faster-whisper 1.2.1/CTranslate2 4.8.5; `uv pip check` clean. Exact tied architecture counts verified: Whisper medium 763,857,920; large-v3 1,543,490,560. Actual seed-42 selection: 100 clips, 705.458813 s total, 21.664875 s max, ID SHA-256 `4469162d...5117`; 500 clips also have zero over 30 s. | No real checkpoint weights were downloaded and no GPU inference/WER result was claimed. New backends are dependency/API-smoked plus mock-tested; live model compatibility, CUDA/cuDNN runtime, and VRAM remain to be established by the resumable run. Phase 05 packaging remains incomplete. | Run `lifemate/asr/run_benchmarks.sh`, compare complete metrics, then promote one ASR checkpoint only if downstream paired agent evaluation also improves. |
| 2026-11-03 | 05 (ChunkFormer cap experiment) | Added an immutable `rnnt_max_symbols_per_frame` setting, cache/fingerprint/report provenance, scoped restoration of the pinned ChunkFormer decoder, cap-1/2/4/64 experiment runner, tests and documentation | Source inspection found the upstream 64-symbol-per-frame internal default. On `train_0664`, cap 64 emitted 227 words; caps 1/2/4 emitted 38/41/47. Five documented chunk/context combinations all retained the 227-word failure. The real cap-2 benchmark path completed that clip with 48.28% WER and recorded the cap. Full suite: 128 passed; Ruff, compileall and shell syntax checks clean. | Upstream `endless_decode()` has no public cap argument, so the adapter uses a lock-protected pinned-version integration. Corpus-wide cap results and downstream semantic effects remain unmeasured. | Run `lifemate/asr/run_chunkformer_caps.sh` on the fixed 500 samples; select a cap only after aggregate and semantic comparison. |
| 2026-11-03 | 05 (ChunkFormer CTC) | Added explicit `chunkformer-ctc` and `chunkformer-rnnt` benchmark identities, checkpoint-family validation, CTC-specific config/report behavior, a paired CTC/RNN-T cap-1/cap-2 runner, selectable single/batch/evaluation ASR engines, tests and docs | Tests verify CTC has no RNN-T cap, never patches RNN-T search, rejects a transducer checkpoint, and is represented completely in evaluation run identity. Full suite: 133 passed; Ruff, compileall and shell syntax checks clean. | CTC checkpoint weights are not present locally, so no live CTC inference or quality/resource result is claimed yet. CLI default remains Qwen; promotion requires benchmark and downstream semantic evidence. | Run `lifemate/asr/run_chunkformer_decoders.sh` on 500 fixed clips and compare complete artifacts, then evaluate the best two engines end to end. |
| 2026-11-03 | 05 (LLM oracle benchmark) | Added a collision-safe Bash wrapper plus model/mode-labeled semantic-evaluation and resume-aware public-batch tqdm progress; suppressed ChunkFormer's nested checkpoint-fetch and per-audio progress bars | Wrapper pins seed, sample count, native mode, controller/token budgets, retry policy and competition timeout to the reused Qwen3.5-4B baseline while exposing URL/model/run label. Existing-output refusal was verified; all progress paths are unit-tested; full suite 135 passed; Ruff, compileall, and `git diff --check` clean. | No new live LLM inference was run. The local folder is named `outputs/llm-bench`, despite the request's `llm-bench-mark` spelling; preserving it avoids splitting the baseline. | Run the 100-sample audio evaluation before the full public batch; compare complete metrics without fitting changes to the train subset. |
| 2026-11-03 | 05 (agent architecture benchmark) | Added versioned grounded prompting, compact invalid-response recovery, a native structured plan-ledger with allowed-tool routing, complete run provenance, a collision-safe four-arm runner, and JSON/Markdown summary generation | Scripted coverage verifies grounded plan evidence, plan-only first request, routed execution tools, compact invalid context, and absence of the internal plan from SDK/submission calls. The runner pins the paired 100-sample protocol and uses uncached ChunkFormer cap 1 in audio mode. Full suite: 140 passed; Ruff, compileall, shell syntax, CLI help, runner usage, and `git diff --check` clean. | No live model benchmark was run and no architecture was selected. The local score remains an approximate strict matcher, so development results require a disjoint confirmation fold. | Run oracle screening first, take the best two to uncached audio, then confirm the winner on a predeclared disjoint fold before another public submission. |

## Phase 00 evidence

- Read the complete local core SDK and every app/helper Python source under `public_test/zaic_tools/`; no SDK file was
  modified.
- Inspected the system catalog entry, all five `x-one-of` schemas, representative `x-ref`/`x-match` entries, two
  history-free input shapes, one history-bearing shape, and the three-row sample submission.
- `scripts/inspect_kit.py` explicitly read no audio/state contents and did not access `train/`.
- Verified source-level behavior is recorded in `docs/official-contract.md` with file/method references.

## Phase 01 evidence

Environment setup:

```powershell
$env:UV_CACHE_DIR=(Join-Path (Get-Location) '.uv-cache')
uv sync
```

Result: CPython 3.13 virtual environment; pytest 8.4.2 and Ruff 0.17.3 installed. Runtime requires Python 3.11+.

Verification:

```powershell
uv run pytest -q
# 12 passed in 0.10s

uv run ruff check lifemate tests scripts
# All checks passed!

uv run python -m lifemate --dataset-root '.\data\train+public_test' `
  --sample-id public_0039 `
  --transcript 'liệt kê báo thức (debug override)' `
  --tool alarm.list --arguments-json '{}' `
  --submission-out '.\outputs\phase01-smoke.jsonl'
```

Sanitized CLI diagnostics: decision source `explicit_actions`, termination `decision_source_done`, one call sent,
step status `read_succeeded`, model usage `null`. The generated row is ignored by git under `outputs/`.

Official schema validation (from `data/train+public_test/public_test/`):

```powershell
& '..\..\..\.venv\Scripts\python.exe' -m zaic_tools `
  --split public_test --root .. `
  --validate '..\..\..\outputs\phase01-smoke.jsonl'
# 1 lines checked, 0 schema errors
```

This establishes schema validity for that one row only. It is not full-split coverage and not semantic success or an
official score.

## Corrected verification issue

The first smoke attempt used inline `--action` JSON. Windows PowerShell native argument handling removed embedded JSON
quotes, so argparse rejected the value and the subsequent validator reported the expected missing output file. A
PowerShell-safe `--tool` / `--arguments-json` CLI path and regression test were added; the full smoke/validator/test/lint
sequence then passed as recorded above.

The first lint command, `uv run ruff check . --exclude 'data'`, also traversed `.venv/` and `.uv-cache/` and failed on
third-party package sources. It was replaced with the repository-scoped command
`uv run ruff check lifemate tests scripts`, which passed. No third-party file was changed.

## Phase 02 evidence

Baseline before Phase 02 edits:

```powershell
uv run pytest -q
# 12 passed in 0.11s

uv run ruff check lifemate tests scripts
# All checks passed!
```

Final verification:

```powershell
uv run python -m compileall -q lifemate tests scripts
uv run pytest -q
# 25 passed

uv run ruff check lifemate tests scripts
# All checks passed!

uv run python -m scripts.phase02_smoke `
  --dataset-root '.\data\train+public_test' `
  --sample-id public_0039 `
  --transcript 'liệt kê báo thức (Phase 02 debug override)' `
  --tool alarm.list --arguments-json '{}' `
  --output '.\outputs\phase02-smoke.jsonl'
# decision_source=llm:scripted:offline-scripted-smoke
# termination_reason=model_completed; tool_calls_sent=1; model requests=2
```

Official validation from `data/train+public_test/public_test/`:

```powershell
& '..\..\..\.venv\Scripts\python.exe' -m zaic_tools `
  --split public_test --root .. `
  --validate '..\..\..\outputs\phase02-smoke.jsonl'
# 1 lines checked, 0 schema errors
```

The smoke used `ScriptedModelClient` and the real SDK. It verifies model-loop plumbing, dynamic permissions, official
validation, ordered execution, and SDK output provenance—not model intelligence or semantic task success.

The first smoke command used `uv run python .\scripts\phase02_smoke.py`; Python placed `scripts\` on `sys.path`, so
the local `lifemate` package import failed and the validator then found no output file. Running the same smoke as the
module `python -m scripts.phase02_smoke` fixed the Windows import path; the corrected command above passed.

## Phase 03 evidence

Baseline before edits:

```powershell
uv run pytest -q
# 25 passed in 0.48s

uv run ruff check lifemate tests scripts
# All checks passed!
```

Verified native Windows environment:

```powershell
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader
# NVIDIA GeForce RTX 5060, 8151 MiB, 3718 MiB, 610.88

$env:UV_CACHE_DIR = '.uv-cache'
uv venv --python 'C:\Users\Tuan\AppData\Roaming\uv\python\cpython-3.12.12-windows-x86_64-none\python.exe' .venv-asr
uv pip install --python .venv-asr\Scripts\python.exe 'qwen-asr==0.0.6'
uv pip install --python .venv-asr\Scripts\python.exe `
  --index-url https://download.pytorch.org/whl/cu130 'torch==2.12.1' --reinstall
```

The first package resolution installed CPU-only `torch==2.15.0`; it was explicitly replaced with PyTorch's official
CUDA 13.0 Windows wheel. Final versions and hardware detection:

```text
Python 3.12.12
qwen-asr 0.0.6
torch 2.12.1+cu130; CUDA runtime 13.0; CUDA available: true
transformers 4.57.6; accelerate 1.12.0
GPU: NVIDIA GeForce RTX 5060; BF16 supported: true
```

The exact stack is now locked as the optional `asr` dependency group:

```powershell
$env:UV_CACHE_DIR = '.uv-cache'
uv lock
# Resolved 126 packages; locked qwen-asr 0.0.6 and torch 2.12.1+cu130

uv pip check --python .venv-asr\Scripts\python.exe
# Checked 94 packages; all installed packages are compatible
```

Real primary-checkpoint end-to-end smoke (local audio, no hosted ASR):

```powershell
$env:HF_HOME = (Resolve-Path .).Path + '\cache\huggingface'
.\.venv-asr\Scripts\python.exe -m lifemate `
  --dataset-root data\train+public_test --sample-id public_0006 `
  --input-mode audio --mode explicit `
  --asr-checkpoint Qwen/Qwen3-ASR-1.7B `
  --asr-backend qwen-asr-transformers --asr-device cuda --asr-dtype bfloat16 `
  --asr-language Vietnamese --asr-max-output-tokens 128 --asr-max-batch-size 1 `
  --no-asr-cache
```

Result: checkpoint downloaded to ignored workspace cache, loaded successfully, and transcribed three turns (final plus
two history audio files). Final-audio inference was 5.607 s on the download/first-use run; combined history inference
was 0.984 s; failures 0. The unchanged runtime created the official session and returned
`decision_source_done`, zero current calls, and a 64-character SDK state hash. No historical call was replayed.

Privacy-safe forced-language comparison:

```powershell
$env:HF_HUB_OFFLINE = '1'
.\.venv-asr\Scripts\python.exe -m scripts.smoke_qwen_asr `
  --sample-id public_0006 --language Vietnamese --limit 2
.\.venv-asr\Scripts\python.exe -m scripts.smoke_qwen_asr `
  --sample-id public_0006 --language auto --limit 2
```

Both runs detected Vietnamese for both fixed clips and produced identical per-clip transcript SHA-256 values and
character counts. Forced run: initialization 13.110 s, final 2.682 s, history 0.487 s. Auto run: initialization
13.054 s, final 2.802 s, history 0.622 s. CUDA memory after initialization was 3893.2 MiB allocated / 4484.0 MiB
reserved; peak allocated was 4480.8 MiB. The script emits hashes/counts rather than private transcript text.

Cache and official output verification:

```powershell
.\.venv-asr\Scripts\python.exe -m lifemate `
  --dataset-root data\train+public_test --sample-id public_0006 `
  --input-mode audio --mode explicit --asr-device cuda --asr-dtype bfloat16 `
  --asr-max-output-tokens 128 --asr-cache-dir cache\transcriptions-smoke-20261101 `
  --submission-out outputs\phase03-smoke.jsonl
# first current-key run: 0/3 hits, 0 failures

# Same command repeated without --submission-out:
# 3/3 hits, hit rate 1.0, inference latency 0, model not loaded

Push-Location data\train+public_test\public_test
& '..\..\..\.venv\Scripts\python.exe' -m zaic_tools `
  --split public_test --root .. --validate '..\..\..\outputs\phase03-smoke.jsonl'
Pop-Location
# 1 lines checked, 0 schema errors
```

End-to-end Phase 02 model-context integration using the successful real Qwen cache and official SDK:

```powershell
$env:HF_HUB_OFFLINE = '1'
.\.venv-asr\Scripts\python.exe -m scripts.phase03_smoke
# decision_source=llm:scripted:offline-scripted-phase03-smoke
# termination_reason=model_completed; model_requests=1
# transcribed_turns=3; cache_hits=3/3; tool_calls=0; state_hash_length=64

# Official validation of outputs\phase03-smoke.jsonl:
# 1 lines checked, 0 schema errors
```

This verifies that actual locally produced Qwen transcripts and reconstructed history enter the Phase 02
`LlmDecisionSource` before the real `ToolEnv` result is captured. It is a plumbing result, not a live-model quality
evaluation.

Final repository verification:

```powershell
.\.venv\Scripts\python.exe -m compileall -q lifemate tests scripts
.\.venv\Scripts\python.exe -m pytest -q
# 37 passed in 0.38s

.\.venv\Scripts\ruff.exe check lifemate tests scripts
# All checks passed!

$env:UV_CACHE_DIR = '.uv-cache'
uv run --frozen pytest -q
# 37 passed in 0.27s
uv run --frozen ruff check lifemate tests scripts
# All checks passed!
```

One intermediate `uv run` attempt after dependency metadata changed failed before running tests because the sandbox
could not access the user-level uv cache/network. Verification continued with the already-installed `.venv`, then
`uv.lock` was updated successfully with approved network access. This was an execution-environment issue, not a test
failure.

There are no remaining Phase 03 acceptance blockers. The public test set has no reference transcripts, so no WER,
CER, semantic task-success rate, or official quality score is claimed. Native-HF and CPU are documented fallback
configurations but were not needed for the successful Windows CUDA run. Phase 04 was not started.

## Phase 04 evidence

Baseline before edits:

```powershell
$env:UV_CACHE_DIR = '.uv-cache'
uv run --frozen pytest -q
# 37 passed in 0.25s
uv run --frozen ruff check lifemate tests scripts
# All checks passed!
```

Official OpenAI documentation was checked before changing provider defaults. It confirms that `gpt-4.1-mini` supports
Chat Completions and function calling, and that the current Chat Completions API exposes `max_completion_tokens` while
`max_tokens` is deprecated. The model page currently marks `gpt-4.1-mini` deprecated; the explicitly requested model
was preserved for initial tests. No OpenAI request was sent.

Final repository verification:

```powershell
$env:UV_CACHE_DIR = '.uv-cache'
uv run --frozen pytest -q
# 48 passed in 0.42s
uv run --frozen ruff check lifemate tests scripts
# All checks passed!
uv run --frozen python -m compileall -q lifemate tests scripts
# Exit 0
```

Tests cover batch order (including parallel out-of-order completion), one session per sample, fresh model conversation,
shared ASR reuse, resume after failure, unsuccessful termination exclusion, duplicate/missing/malformed submission
rejection, SDK-derived rows, scripted model batch operation, transient API backoff, non-transient errors, shared request
rate limiting, current OpenAI token-field selection, and existing single-sample behavior.

Three-sample real-SDK smoke and resume:

```powershell
uv run --frozen python -m scripts.phase04_smoke
# [1/3] ... [3/3]
# selected=3, completed=3, failed=0, ready_for_submission=true
# metrics official_schema_validation: 3 lines checked, 0 schema errors

uv run --frozen python -m scripts.phase04_smoke --resume
# completed=3, failed=0; no sample was rerun; official validation passed again
```

Full offline coverage run—scripted model only, no ASR/model network/API cost:

```powershell
uv run --frozen python -m scripts.phase04_smoke `
  --full --output '.\outputs\phase04-full-offline\submission.jsonl'
# 634/634 completed; 0 failures; ready_for_submission=true; wall command 13.6 s

uv run --frozen python .\scripts\verify_submission.py `
  --dataset-root '.\data\train+public_test' `
  --submission '.\outputs\phase04-full-offline\submission.jsonl'
# Expected samples: 634, submission rows: 634
# PASS: structurally valid with every expected ID
```

`metrics.json` records `dataset_samples=634`, `completed_samples=634`, `failed_samples=0`,
`ready_for_submission=true`, and official validation `634 lines checked, 0 schema errors`; `failures.jsonl` is empty.
Every hash and call list came from its sample's real official `ToolEnv`. All scripted decisions intentionally returned
`unsupported`, so this proves batch/session/output plumbing only and is not a useful competition submission or score.

No full paid OpenAI batch, real-model semantic evaluation, Phase 05 optimization, deployment, or external submission
was performed.

## Phase 04.5 evidence

Baseline before edits:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run --frozen pytest -q
# 51 passed in 0.69s
uv run --frozen ruff check lifemate tests scripts
# All checks passed!
```

The installed `qwen-asr==0.0.6` API was inspected before deadline/decoding changes. Its `transcribe()` signature has no
timeout or generation-kwargs parameter, and its Transformers path calls synchronous `model.generate()` internally.
The wrapper's underlying Transformers model now receives explicit `do_sample=False` and `num_beams=1`; deadline
overruns are detected after return. The native-HF path additionally receives remaining time through `max_time`.

Final verification:

```powershell
$env:UV_CACHE_DIR='.uv-cache'
uv run --frozen pytest -q
# 60 passed in 0.51s
uv run --frozen ruff check lifemate tests scripts
# All checks passed!
uv run --frozen python -m compileall -q lifemate tests scripts
# Exit 0
```

Focused compliance tests account for nine of the 60 tests and cover mocked LLM/ASR overruns, diminishing multi-step
budgets, request-timeout propagation/retry suppression, deterministic seed setup, explicit greedy decoding, exactly
one ToolEnv, competition validator suppression, prohibited-path scanning, failed/successful serialization, and old/new
CLI defaults. They require no GPU or external API.

Real-SDK competition-mode single sample:

```powershell
uv run --frozen python -m lifemate --competition-mode `
  --sample-id public_0039 --input-mode text --transcript 'debug list alarms' `
  --mode explicit --tool alarm.list --arguments-json '{}' `
  --submission-out 'outputs/phase045-competition-single.jsonl'
# decision_source_done; one alarm.list call; SDK-derived 64-character state hash; exit 0
```

Backward-compatible development validation:

```powershell
uv run --frozen python -m scripts.phase04_smoke `
  --output 'outputs/phase045-development-smoke/submission.jsonl'
# 3/3 completed; 0 failures; ready_for_submission=true; official validator passed
```

Outer watchdog checks:

```powershell
uv run --frozen python -m scripts.competition_watchdog --timeout-seconds 5 -- `
  .\.venv\Scripts\python.exe -c "print('watchdog-ok')"
# watchdog-ok; exit 0

.\.venv\Scripts\python.exe -m scripts.competition_watchdog --timeout-seconds 0.2 -- `
  .\.venv\Scripts\python.exe -c "import time; time.sleep(2)"
# sample_timeout emitted; child stopped in approximately 0.33 s wall time
```

The watchdog is the hard-stop mechanism for blocking CUDA calls; Python deadline checks alone are not described as
cancellation. Phase 05 must provide the fully offline local LLM, export pre-interpreter environment settings, and wire
one watchdog child per scored sample. No external API, model download, or real ASR inference was run for Phase 04.5.

## Phase 04.6a evidence — synthetic completion and bounded duplicate recovery

Implemented on 2026-11-02 after inspecting the installed SDK lifecycle and trace behavior again. The official
`Session.call` appends every dispatched call before execution, while `Session.validate` records nothing. Therefore
`agent.finish`, invalid finish actions, and locally rejected duplicates are intercepted before `Session.call`; actual
SDK calls remain sourced unchanged from `env.get_tool_calls()` and `env.get_state_hash()`.

Behavior changes:

- provider-safe `agent__finish` maps to internal `agent.finish`; valid `done`/`unsupported` calls terminate locally;
- free-text JSON completion is rejected and bounded by the existing invalid-proposal limit;
- multi-call responses are rejected atomically and execute no bundled operation;
- confirmed duplicate writes produce `duplicate_write_rejected` feedback and one corrective model turn;
- uncertain writes are never retried, distinct writes remain executable, and explicit identical repeats are allowed
  conservatively when the transcript says twice/repeat;
- normal CLI and batch inference now default to a 28-second whole-sample cooperative deadline, with remaining-time
  HTTP caps and the existing 30-second per-sample watchdog as the non-cooperative hard stop;
- failures/checkpoints include per-request tokens, response category, selected tool, remaining budget, controller/SDK
  step outcomes, ASR initialization separately from inference, and phase timings. Raw malformed responses require
  `--capture-raw-model-responses`.

Verification run from the repository root:

```powershell
uv run --frozen ruff check lifemate tests
# All checks passed!

uv run --frozen pytest -q
# 72 passed in 2.38s

uv run --frozen pytest -m integration -q
# 6 passed, 66 deselected in 0.07s
```

The full suite includes real public-test SDK integration tests and confirms synthetic actions are excluded from the
official trace.

Live deterministic Qwen regressions were then run against the healthy local llama.cpp endpoint at
`http://127.0.0.1:1990/v1`, each inside the 30-second watchdog, with the original 0.6B ASR cache/configuration:

- `public_0002`: `read_succeeded -> write_succeeded -> agent_finished`; 3 model requests, 24,146 total tokens,
  approximately 10.5 seconds command wall time. The official trace contains only quote/book calls.
- `public_0005`: three reads, one successful write, another read, then `agent_finished`; 6 model requests, 58,784
  total tokens, approximately 11.9 seconds wall time. No duplicate write reached the SDK.
- `public_0006`: `read_succeeded -> agent_finished`; 2 model requests, 10,694 total tokens, approximately 7.8 seconds
  wall time. The official trace contains only the read.

All three terminated as `model_completed`; no synthetic finish appeared in any submission trace. These runs prove the
termination, trace, and budget regressions, not semantic task accuracy. Cached ASR text remains noisy, and in
particular `public_0006` may have finished before performing the intended conditional action.

## Phase 04.6b evidence — trial-01 coverage hardening

Trial 01 produced 596 semantic completions and 38 failures. Its dominant failures were 17 malformed-output limits,
10 timeouts, 8 tool-call-budget exhaustions, 2 duplicate-recovery exhaustions, and only one max-step exhaustion. The
saved manifest recorded 20 model steps and 20 SDK calls; several failures repeated one read tool all 20 times, while
`public_0367` alternated malformed text and the same `recipes.search` call for 20 turns.

The controller now defaults to required native tool calls, keeps a cumulative malformed-output count, rejects
identical unchanged-state reads, leaves a corrective finish turn at the SDK-call limit, and writes SDK-derived degraded
rows for agent-level failures. No local rejection enters the official trace.

Verification:

```powershell
uv run --frozen ruff check lifemate tests
# All checks passed!

uv run --frozen python -m compileall -q lifemate tests scripts
# Exit 0

uv run --frozen pytest -q
# 77 passed in 2.29s

uv run --frozen pytest -m integration -q
# 6 passed, 71 deselected in 0.07s
```

Live llama.cpp checks with trial-02 settings:

- `public_0367`: one request, `sys.ask_missing_info`, `awaiting_user`; the previous 20-turn malformed/read loop did not
  recur.
- `public_0213`: 12 permitted SDK calls, then a locally rejected 13th proposal and `agent.finish`; termination changed
  from `tool_call_budget_exhausted` to `model_completed`, with exactly 12 official calls in the trace.

These are controller/coverage checks, not claims that the selected tool sequences are semantically optimal.

## Phase 04.6c evidence — fail-soft submission coverage

Batch exceptions now retain a structurally valid row whenever the official SDK is usable. If a session already
exists, the runner snapshots that same session so every recorded tool call is preserved. If ASR or controller setup
fails before runtime, it creates one empty SDK session and records its authoritative empty trace and state hash.
Failures remain visible in `failures.jsonl` and are retried by `--resume`; `ready_for_submission` describes structural
coverage rather than semantic success. Only an SDK creation/snapshot failure can still leave `submission` null.

Verification: `uv run --frozen pytest -q` → 79 passed; `uv run --frozen ruff check .` → all checks passed;
`git diff --check` → clean. Regression tests cover both a pre-runtime empty fallback and a post-tool crash that reuses
the existing session without duplicating ToolEnv construction.

## Phase 04.6d evidence — labeled scoring and output-mode A/B review

The evaluator uses the user-supplied empty-gold/hash-gated call-F1 formula. Because the kit documents fuzzy
`x-match` behavior but does not ship the organizer's call-equivalence algorithm, results remain explicitly named
`challenge_score_estimate` and use strict canonical unordered one-to-one matching. Gold full-state hashes come from
isolated SDK replay after inference; train labels never enter the planner.

Review on 2026-11-03 found and fixed five implementation mismatches:

- nested `MappingProxyType` diagnostics could still fail JSON export even after only the outer mapping was copied;
- evaluator deadlines reached `Runtime` but not `LlmDecisionSource`, so HTTP timeout/retry bounds could exceed the
  remaining sample budget;
- constrained-JSON requests sent `tools=[]`, `tool_choice=required` and `parallel_tool_calls=false` alongside
  `response_format`, an avoidable provider compatibility conflict;
- empty `agent.finish` arguments were silently normalized to a successful finish, weakening the documented schema;
- evaluation defaults/provenance and latency did not expose the ASR checkpoint or cache-order bias.

The writer now recursively converts read-only mappings, evaluation sets the same deadline on the decision source,
tool controls are emitted only for native requests with tools, finish arguments remain strict, evaluation defaults to
the competition-compatible 0.6B ASR, and metrics include run identity plus model-only and ASR/cache latency fields.
The A/B root now receives `evaluation-config.json` before inference, while every mode directory receives
`run-config.json` before its sample loop. Both are sanitized and never contain API-key values.

Saved 100-sample A/B evidence (seed 42):

| Input | Mode | Estimate | Hash match | Perfect | Partial | Tool name | Argument |
|---|---|---:|---:|---:|---:|---:|---:|
| oracle | native | 35.3960 | 0.57 | 0.21 | 0.24 | 0.6098 | 0.4773 |
| oracle | constrained JSON | 22.9167 | 0.36 | 0.19 | 0.06 | 0.3933 | 0.3143 |
| ASR | native | 12.9359 | 0.30 | 0.08 | 0.09 | 0.4045 | 0.2414 |
| ASR | constrained JSON | 10.7333 | 0.27 | 0.08 | 0.05 | 0.3100 | 0.2146 |

Native therefore remains the selected output mode. The ASR score delta is meaningful for this paired sample, but
the saved ASR wall-time comparison is not: native had 0/333 cache hits and 228.16 seconds of ASR inference, while
constrained JSON had 333/333 hits. The saved artifacts predate run-identity recording, so the exact checkpoint cannot
be audited from them; future reports record it. These train estimates are diagnostic only and must not be optimized
as if they were private-test truth.

Final verification:

```bash
UV_CACHE_DIR=/tmp/lifemate-uv-cache uv run --frozen pytest -q \
  tests/test_evaluation.py tests/test_agent.py tests/test_model.py \
  tests/test_compliance.py tests/test_sdk_integration.py
# 81 passed in 2.58s

UV_CACHE_DIR=/tmp/lifemate-uv-cache uv run --frozen pytest -q
# 120 passed in 2.41s

UV_CACHE_DIR=/tmp/lifemate-uv-cache uv run --frozen ruff check lifemate tests scripts
# All checks passed!
```
