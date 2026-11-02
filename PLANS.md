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
| 04.6 | Agent termination + duplicate recovery | Synthetic finish protocol, bounded duplicate recovery, default whole-sample budget, diagnostics | COMPLETE (2026-11-02) |
| 05 | Optimize + packaging | Repeatable experiment comparisons, selected configuration, private packaging only once rules verified | TODO |

**Implementation order:** 00 → 01 → 02 → 03 → 04 → 04.5 → 05. You may implement the batch writer earlier if needed; keep milestone acceptance independent.

## Decision register
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
- **Unverified:** exact scoring details/weights, model/API/network rules, and the private Docker execution contract.
- **Next:** Phase 05 must implement/package the fully offline local LLM and wire the per-sample outer watchdog into the organizer-facing entrypoint. Do not run paid full-scale inference without authorization.

When local source contradicts this document, update the decision register and cite the official file/line in docs instead of silently following assumptions.
