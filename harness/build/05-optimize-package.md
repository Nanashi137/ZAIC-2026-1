# Phase 05 — Optimize and prepare reproducible package

Dependency: Phase 04.5 compliance controls must remain enabled. The final organizer-facing entrypoint must use the
per-sample outer watchdog and a fully offline local LLM backend.

## Deliverables
- Standalone fixed-subset ASR benchmark for Qwen3-ASR 0.6B/1.7B, PhoWhisper medium/large, Transformers Whisper
  medium/large-v3, faster-whisper medium/large-v3, ChunkFormer-RNNT-large-vie and ChunkFormer-CTC-large-vie. Record WER, CER, latency/RTF,
  initialization, exact integer model parameters with provenance, and GPU peak memory;
  use progress reporting plus per-success checkpoint/resume. Whisper-family inputs must be preflighted against the
  30-second window and never silently chunked. Evaluate ChunkFormer RNN-T per-frame emission caps 1/2/4 against a
  cap-64 control using separate immutable output identities before promoting it to the production CLI.
  Compare the surviving RNN-T cap-1/cap-2 candidates with an explicitly identified CTC backend on the identical
  selected clips; reject checkpoint/backend family mismatches during initialization. Keep Qwen as the CLI default
  while allowing explicit ChunkFormer engines for downstream semantic A/B runs with fully logged ASR identity.
- Versioned experiment configurations and compare runs on a fixed development subset; track tokens, wall time, valid call rate, no-ops, and error categories. Only use official score when available.
- A collision-safe LLM oracle benchmark wrapper fixes native mode, the 100-sample seed-42 subset, controller budgets,
  and competition timeout while allowing only the local server URL, model name, and output label to vary.
- A separate agent benchmark keeps the direct baseline reproducible and compares grounded prompting, compact invalid
  recovery, and an internal plan-ledger architecture on paired oracle or uncached audio inputs.
- Optimize likely errors: poor transcription, prompt verbosity, tool routing, `x-ref` IDs, clarification, context/history representation and premature stopping.
- Pin dependencies and write Windows setup + run instructions for a fresh machine.
- Inspect **new official documentation** before implementing private-test Docker packaging. Do not invent private submission requirements or assume external LLM APIs are allowed.

## Acceptance
- The ASR benchmark runs independently of the agent/SDK, records immutable selection/config hashes, and can resume
  after a clip or process failure without duplicating successful records.
- A clean local environment can reproduce an example and batch submission when dependencies/configuration are supplied.
- Tests and schema validation pass; experiment comparisons show measured evidence rather than narrative claims.
- Private-test packaging, if required, follows official rules and is validated separately.

## Current evidence (2026-11-03)

- Backend package refactor preserves the existing `lifemate.asr` import surface and Qwen cache identity behavior.
- Unit tests exercise PhoWhisper's pre-load duration rejection, ChunkFormer result parsing, common Vietnamese text
  normalization, the ChunkFormer RNN-T cap and restoration behavior, per-success checkpointing, one-failure
  continuation, pending-only resume and fully-complete resume.
- The full suite passes (140 tests), Ruff/compileall/diff checks pass, and the locked real dependencies import:
  torch 2.12.1+cu126, torchaudio 2.11.0+cu126, chunkformer 1.2.2, faster-whisper 1.2.1 and CTranslate2 4.8.5.
  `uv pip check` reports no conflicts.
- Fresh read-only preflight of the seed-42 100-final-clip subset reports 705.458813 seconds total and 21.664875
  seconds maximum, so the current subset fits the explicit 30-second PhoWhisper no-chunking protocol.
- The all-model runner accepts a positive sample limit as its third positional argument (default 100), allowing the
  same protocol to scale to 500 samples without editing the script.
- Every report now requires an exact parameter count. Loaded PyTorch models are counted through unique parameters;
  CTranslate2/faster-whisper uses a tied meta-model instantiated from the explicitly recorded canonical OpenAI
  Transformers config, so quantization/runtime format does not replace the logical architecture count.
- The user-completed fixed 500-sample run provides real WER/CER, latency and VRAM artifacts for seven backends;
  ChunkFormer-RNNT led normalized WER at 22.57% and exposed the decoder failure described below. The two
  faster-whisper directories contain configuration only and are not treated as completed evidence.
- The completed 500-sample artifacts identified one ChunkFormer RNN-T runaway digit emission. Inspection of the
  pinned 1.2.2 source found an internal 64-symbol-per-frame default that the public `endless_decode()` API does not
  expose. A local CPU reproduction reduced the failing output from 227 words at cap 64 to 38/41/47 words at caps
  1/2/4 respectively. Changing documented encoder chunk/context sizes did not alter the 227-word failure. The full
  paired cap experiment is still pending and must choose the cap by aggregate and downstream semantic evidence.
- `scripts/run_llm_oracle_benchmark.sh` writes each model beneath `outputs/llm-bench/<run-label>` and refuses to
  overwrite existing evidence. Semantic evaluation displays a tqdm bar for the selected samples with the model and
  mode in its label. The reused Qwen3.5-4B baseline remains untouched; no additional live LLM run was performed
  during wrapper implementation.
- Public-test batch inference displays resume-aware tqdm progress initialized from validated checkpoint coverage;
  updates occur only after each sample's atomic checkpoint is written. ChunkFormer's internal one-file tqdm is
  suppressed by a scoped override so it cannot emit a separate `0/1` line for every sample; the same scope silences
  its Hugging Face checkpoint-fetch bar without hiding warnings or errors.
- Agent experiment flags and run identity cover architecture, prompt version, invalid-response history, and retry
  policy. Plan-ledger is native-only, validates plan evidence against the final transcript, restricts execution tools
  to the plan, and is intercepted before SDK validation. Scripted tests prove the plan is absent from SDK/submission
  calls. `scripts/run_agent_benchmark.sh` runs four collision-safe paired variants; audio mode explicitly disables
  the transcription cache. No live architecture benchmark has been run.
