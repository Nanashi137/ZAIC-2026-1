# Roadmap and checkpoints

Codex: work one phase at a time, using `harness/build/<phase>.md`. Do not mark complete until the stated verification has actually run. Update this plan and `harness/build-log.md` after each phase.

| Phase | Objective | Exit criterion | Status |
|---|---|---|---|
| 00 | Inspect installed official SDK and validate assumptions | `docs/official-contract.md` grounded in full local source and kit inspection report | COMPLETE (2026-10-30) |
| 01 | Data reader, schema selection, environment adapter, text-only runner | One real public-test sample session produces `id/tool_calls/state_hash`; unit + SDK smoke tests | COMPLETE (2026-10-30) |
| 02 | LLM-driven multi-step tool agent | Validated read→write tool loop, safe failure/clarification handling, tests | COMPLETE (2026-10-30) |
| 03 | Qwen3-ASR + conversation history | Audio and audio history reach agent correctly; reproducible ASR tests | COMPLETE (2026-11-01) |
| 04 | Batch submission + diagnostics | Dynamic sample coverage, structural validator and official SDK schema validation | COMPLETE (2026-11-01) |
| 04.5 | Competition compliance + runtime hardening | 28 s competition deadline, deterministic controls, one-ToolEnv scored path, compliance tests | COMPLETE (2026-11-01) |
| 04.6 | Agent/runtime hardening + labeled mode evaluation | Finish/duplicate/read-loop controls, fail-soft rows, native/constrained-JSON oracle/ASR evaluation, isolated full-state gold replay, challenge-score estimate, paired diagnostics, JSON-safe export and auditable run identity; 120 tests passed | COMPLETE (2026-11-03) |
| 05 | Optimize + packaging | Repeatable experiment comparisons, selected configuration, private packaging only once rules verified | IN PROGRESS (2026-11-03) |

**Implementation order:** 00 → 01 → 02 → 03 → 04 → 04.5 → 04.6 → 05. Keep milestone acceptance independent.

## Decision register
- **Confirmed:** Phase 04.6 (2026-11-02) includes user-authorized labeled-train offline scoring. Primary
	`challenge_score_estimate` applies the supplied empty-gold/hash-gated F1 formula over all evaluated samples.
	Exact call matching is unavailable: deterministic strict unordered one-to-one fallback is explicitly approximate.
	Gold hashes use isolated offline SDK reference replay after inference, cached across modes; hashing partial
	`label.final_state` directly was incorrect and has been removed. No gold leaks into inference. Constrained JSON
	now receives tool/finish schemas with action-specific output constraints. Error rates distinguish malformed
	output, local schema/finish rejection, abnormal termination and missing results. Focused agent/evaluation/CLI/
	compliance/SDK tests: 69 passed; subsequent export regressions reproduced and fixed read-only diagnostic
	mappings reaching JSON serialization (both modes, raw capture on/off: 4 passed). Review on 2026-11-03 fixed
	deep read-only mapping serialization, evaluation-to-model deadline propagation, constrained requests carrying
	conflicting native tool controls, invalid empty finish acceptance, ASR provenance/defaults, and cache-biased
	A/B latency reporting. Evaluation now writes a sanitized root config before inference and a per-mode config before
	each sample loop; API-key values are excluded. Full suite: 120 passed.
	Phase 05 is not advanced by this work.
- **Confirmed:** tool execution uses official `ToolEnv`; actual calls and state hash are SDK-provided.
- **Confirmed:** initial environment already includes previous history tool effects.
- **Confirmed:** `sys.ask_missing_info` is always allowed, requires `tool_name` + `params`, records no state, and returns `waiting_for_user`.
- **Confirmed:** the SDK validator enforces `x-one-of`; `x-ref` and `x-match` are annotations, not validation rules.
- **Confirmed:** Phase 01 uses Python 3.11+, `uv`, a single sequential runtime, and no model/network/GPU dependency.
- **Confirmed:** Phase 02 provides a configurable OpenAI-compatible sequential model loop, dynamic permitted-tool schemas, actual observation feedback, and deterministic offline tests.
- **Confirmed:** non-local model endpoints require explicit `--allow-external-data`; no external provider was called during Phase 02 verification.
- **Confirmed:** Phase 03 uses only local Qwen3-ASR (`Qwen/Qwen3-ASR-1.7B`) through `qwen-asr`'s Transformers backend; there is no faster-whisper fallback.
- **Confirmed:** native Windows 11 inference works on the RTX 5060 with Python 3.12, PyTorch CUDA 13.0, and BF16; the final request and ordered user-audio history feed the unchanged Phase 02 runtime.
- **Confirmed:** assistant history text/tool calls/results remain context-only, historical calls are not replayed, and content-addressed transcription caching is ignored by version control.
- **Confirmed:** Phase 04 provides deterministic resumable batch inference, per-sample model/session isolation, shared ASR/client resources, incremental checkpoints, diagnostic metrics/failures, and atomic final export only after exact coverage checks.
- **Confirmed:** a 634-sample offline scripted-model run produced SDK-derived rows in input order; the structural checker passed and the official validator reported 634 lines with zero schema errors. This is plumbing evidence, not model quality.
- **Confirmed:** the live provider defaults requested for testing are OpenAI Chat Completions at `https://api.openai.com/v1` with `gpt-4.1-mini`; remote data transfer still requires explicit `--allow-external-data`, and no paid API call was made during Phase 04.
- **Confirmed:** Phase 04.5 competition mode defaults to a 28-second monotonic sample budget, disables in-process post-batch official validation, rejects non-local model endpoints, seeds available Python/NumPy/PyTorch/CUDA runtimes, and explicitly configures greedy ASR decoding.
- **Confirmed:** cooperative deadline checks and model request timeout/retry caps cannot interrupt the official `qwen-asr` synchronous CUDA call. `scripts/competition_watchdog.py` is the required outer process boundary for a hard stop; Phase 05 packaging must invoke it once per scored sample.
- **Confirmed:** production modules do not construct, probe, read, or parse `<split>_environment` paths and expose no `get_state()` call. Only the organizer's `ToolEnv` loads app state. Competition mode creates one inference ToolEnv and skips the validator that would create another.
- **Confirmed:** model completion uses the internal `agent.finish` action (provider-safe name `agent__finish`), which is intercepted before official validation/dispatch and never appears in the SDK trace or submission.
- **Confirmed:** an SDK-confirmed duplicate write is rejected locally and receives at most one corrective model turn; uncertain writes still stop without retry. Explicitly requested identical repetition is conservatively recognized from the transcript.
- **Confirmed:** normal single-sample and batch CLI runs now default to a 28-second monotonic cooperative deadline, capped below the 30-second per-sample watchdog boundary. Per-request token usage and phase timings are checkpointed; raw malformed responses are opt-in.
- **Confirmed:** Phase 04.6 defaults native model requests to `tool_choice=required`, counts malformed responses cumulatively across valid calls, rejects identical reads without an intervening write, and leaves one corrective turn after the official call budget so `agent.finish` remains available.
- **Confirmed:** every recoverable batch failure produces a degraded, SDK-derived submission row. Failures after session creation preserve that session's complete trace/hash; failures before runtime create one empty SDK session and use its authoritative empty trace/hash. Metrics still distinguish semantic failures from structural submission coverage. Only SDK session creation/snapshot failure can leave coverage incomplete.
- **Confirmed:** Phase 05 starts with an agent-independent ASR benchmark under `lifemate/asr/`. It uses one fixed
  100-final-clip train subset (seed 42), common normalization, duration preflight, tqdm progress, fsync'd per-success
  checkpoints and strict resume fingerprints. Qwen 0.6B/1.7B, PhoWhisper medium/large, Transformers and
  faster-whisper medium/large-v3, and ChunkFormer RNN-T/CTC share the same result contract. Whisper-family models never
  silently chunk; a selected clip over 30 seconds fails preflight. Every report requires an exact integer parameter
  count and records whether it came from the loaded model or a tied canonical architecture config.
- **Confirmed:** the 500-sample ChunkFormer-RNNT run exposed one runaway digit emission. Pinned `chunkformer==1.2.2`
  permits 64 non-blank emissions per acoustic frame internally, while its documented `endless_decode()` API exposes
  no corresponding cap. The adapter now records and enforces `rnnt_max_symbols_per_frame` with a scoped,
  lock-protected override, includes it in cache/run identity, and provides a paired cap-1/2/4/64 runner. A one-clip
  reproduction supports the mechanism but does not select the final cap; full benchmark and downstream agent A/B
  evidence remain required.
- **Confirmed:** ChunkFormer CTC is now an explicit benchmark backend rather than a checkpoint-only alias. It records
  `backend=chunkformer-ctc`, omits the RNN-T emission cap, bypasses the scoped RNN-T decoder override, and validates
  the loaded checkpoint's `config.model=asr_model`; RNN-T reciprocally requires `config.model=transducer`. A paired
  fixed-subset runner compares CTC with RNN-T caps 1 and 2. Live CTC WER/latency/VRAM and downstream semantic score
  remain pending. Single, batch and evaluation CLIs retain Qwen as default but can explicitly select either
  ChunkFormer backend; evaluation run identity records the engine and decoder settings.
- **Confirmed:** Phase 05 LLM comparison uses a fixed 100-sample seed-42 oracle/native protocol matching the saved
  Qwen3.5-4B baseline. A collision-safe wrapper varies only the local base URL, model name, and run label, logs the
  evaluator's sanitized root/per-mode configurations, displays model/mode-specific tqdm sample progress, and refuses
  to overwrite an existing model directory under `outputs/llm-bench`.
- **Confirmed:** public-test batch inference uses a resume-aware tqdm bar. Its initial count comes from validated
  checkpoint records and each update follows the atomic sample checkpoint, without changing submission semantics.
  ChunkFormer's nested one-file tqdm is suppressed through a scoped, lock-protected module override so only the
  aggregate evaluator/batch bar is rendered.
- **Confirmed:** Phase 05 agent experiments are explicit run configuration, not a silent baseline change. The direct
  baseline remains the default; `grounded-v2`, compact invalid-response history, and native-only plan-ledger can be
  selected independently. Plan-ledger validates transcript evidence and allowed candidate tools, routes subsequent
  model calls to planned tools plus clarification/finish, and never dispatches its internal plan to the SDK. A
  collision-safe four-arm 100-sample runner supports oracle or uncached ChunkFormer-cap-1 audio and emits a paired
  summary. No live result or winning architecture is claimed yet.
- **Confirmed:** confirmation experiments use a persisted 100-sample fold selected with seed 314159 after excluding
  the complete seed-42 100-sample development fold. The confirmation runner stores selection provenance under
  `outputs/agent/<experiment-name>`, compares only the three direct variants, keeps audio uncached, and reports
  paired sample changes plus a deterministic bootstrap interval. No confirmation result is claimed before execution.
- **Confirmed:** model requests can now be reproducibly seeded per sample turn with a stable SHA-256 derivation from
  run seed, sample ID, and request index. The seed is sent in the chat-completion payload and logged in request
  diagnostics; run identity can also record local model artifact path/name/size/SHA-256 and quantization. The Q4
  oracle A/A runner writes collision-safe paired evidence under `outputs/a-a_test` and reports first-action and full
  trace agreement in addition to score deltas.
- **Confirmed:** the live Q4_K_M seeded oracle A/A run on the seed-314159 confirmation fold scored 25.7525 and
  25.9889 (paired delta +0.236 points, bootstrap interval [-4.123, 4.461]). All 100 first-request seeds and every
  shared request-index seed matched, but only 79/100 first actions and 42/100 complete traces matched. Request-level
  seeds stabilize the aggregate estimate but do not make the current llama.cpp CUDA backend bitwise/trajectory
  deterministic; agent experiments must retain paired intervals and treat small deltas cautiously.
- **Rejected:** domain routing and the follow-up intent ledger were removed after paired oracle evaluation. Router v2
  scored 24.52 versus 24.88 direct (delta -0.362, interval [-8.207, 7.658]) and left 25/91 actionable samples without
  complete domain coverage. Intent-ledger scored 21.65 versus 24.88 direct (delta -3.233, interval [-11.071, 4.697]);
  23/100 samples never produced a valid ledger, 95 gold calls were missing, and invalid-output rate reached 36%.
  Persisted output evidence remains under `outputs/agent`, but both controllers, their CLI choices, and runners were
  reverted. Future work must not put a mandatory global plan/router gate in front of execution.
- **Unverified:** official call equivalence/assignment (including fuzzy `x-match`), model/API/network rules,
  and the private Docker execution contract. Scoring formula supplied by the user; exact matcher remains unavailable.
- **Next:** Phase 05 must implement/package the fully offline local LLM and wire the per-sample outer watchdog into the organizer-facing entrypoint. Do not run paid full-scale inference without authorization.

When local source contradicts this document, update the decision register and cite the official file/line in docs instead of silently following assumptions.
