# Phase 03 — Qwen3-ASR and conversation history

Status: COMPLETE on 2026-11-01. Phase 04 was not started.

## Selected implementation

- Primary checkpoint: `Qwen/Qwen3-ASR-1.7B`.
- Primary backend: official `qwen-asr==0.0.6` Transformers wrapper.
- Optional checkpoint: `Qwen/Qwen3-ASR-0.6B` through the same backend and configuration surface.
- Optional alternate backend: native Transformers with a matching `Qwen3-ASR-*-hf` checkpoint only; backend and checkpoint families cannot be mixed.
- No faster-whisper dependency or fallback, hosted ASR API, fine-tuning, forced aligner, timestamps, FlashAttention 2, or vLLM.

## Delivered behavior

- `Qwen3ASRTranscriber` validates public-split audio paths, passes paths directly to Qwen, loads once, reuses the model, preserves detected language, records latency, and raises typed initialization/path/inference errors.
- Checkpoint, revision, backend, device, dtype, language, token limit, and batch size are configurable. Automatic placement chooses CUDA when available and CPU/float32 otherwise.
- The audio pipeline transcribes the final request first and prior user audio in original history order. It preserves assistant text and recorded tool calls/results as context, then invokes the unchanged Phase 02 runtime.
- Historical tool calls are never submitted to `ToolEnv`. The runtime still creates exactly one official session for current-turn inference and derives output only from `get_tool_calls()` and `get_state_hash()`.
- Optional content-addressed cache keys include audio SHA-256, checkpoint/revision, backend, language, max tokens, dtype, and preprocessing version. Only successful results are cached under ignored `cache/` storage.
- Text-debug inference remains available through `--input-mode text`; real audio uses `--input-mode audio`.

## Verified acceptance

- Fake-ASR tests cover history/no-history, ordering, preservation, replay prevention, path errors, initialization/model reuse, CPU/CUDA selection, inference failures, cache identity/failures, and text-only regression.
- A real SDK integration test uses `public_0006` with final plus two historical audio references and records no historical tool calls.
- Real native-Windows inference succeeded with the 1.7B checkpoint on all three clips for `public_0006`.
- Forced Vietnamese and automatic detection were compared on a fixed two-clip set; both detected Vietnamese and produced identical transcript SHA-256 values. No reference transcripts exist, so no WER/CER or quality score is claimed.

See `harness/build-log.md` for exact commands, versions, timing, memory, cache, and official SDK evidence.
