# LifeMate — repository instructions for Codex

## Mission
Build a small, testable, **competition-compliant** Vietnamese voice-to-action agent for Zalo AI Challenge 2026 LifeMate. The priority is correct official tool execution and a reproducible submission, **not** a demo app or a large agent framework.

Read `GOALS.md`, `PLANS.md`, and `docs/official-contract.md` before substantial changes. The *installed official kit* is the final authority if these notes disagree with it. `PROMPTS.md` contains reusable phase prompts.

## Environment and paths
- Developer machine: **Windows 11, PowerShell**, VS Code/Codex.
- Dataset root (directory containing `public_test/`): `data/train+public_test/` relative to repo root. Make this path configurable; do not assume CWD or use Linux-only shell commands.
- Make split names configurable in application code so a later private split can use the same pipeline; develop and test **only** on `public_test/` for now. **Do not inspect, use, index, copy, modify, or train on `train/`** unless the user explicitly changes scope.
- Dataset and official SDK are third-party/private competition materials. Do not commit, redistribute, reformat, or alter them. Never upload them, audio, states, prompts, or traces to an external service without checking user approval and competition rules.
- Prefer Python 3.11+, `uv`, pytest, Ruff, Pydantic v2 only where they add value, and a configurable OpenAI-compatible model client. ASR is an optional later phase.

## Must-follow official runtime contract (from toolkit README v3.2.0, verify locally)
- Input: one JSONL record per sample: `id`, `audio`, `context`, `history`. `context` includes sample-specific time/timezone/profile/location and `installed_apps` with allowed tools. History's prior actions **have already been applied to initial environment state**; do not replay them.
- User history turns may contain additional audio. The *last user's audio* is the action to solve. History must be interpreted, not replayed.
- Tool names, descriptions, parameters, kinds, and extensions (`x-one-of`, `x-ref`, `x-match`) come from `<split>_tools_catalog.json`. **Restrict offered/selected tools to the sample's allowed tools** (and inspect the system/meta tool rules in SDK).
- Create **exactly one** `ToolEnv(split, sample_id, root=dataset_root)` *per sample inference*. Do not reset/recreate it to retry a plan, even after a schema/tool error.
- Before execution use `env.validate(name, arguments)`; it checks schema but does not execute. `env.call(name, arguments)` executes and returns an observation. Invalid calls are recorded by the SDK as attempts, so prevent them where possible.
- After inference, use **only** `env.get_tool_calls()` for the ordered recorded calls and `env.get_state_hash()` for the output hash. Never fabricate either. Calls rejected by the SDK may appear in `get_tool_calls()`; do not filter them out.
- Do **not** read `<split>_environment/<id>/state.json` or `env.get_state()` as input to agent planning. Read state through permitted tools. Offline diagnosis/SDK validation is separate and must never feed privileged state to the agent.
- An SDK tool may return `{"ok": true, "done": false}` for a no-op. This is **not** evidence that the requested action succeeded; inspect observations.
- Submission format: one UTF-8 JSONL line per sample with `id`, `tool_calls`, `state_hash`. `tool_calls=[]` is valid for an out-of-scope request; the missing-info system tool must follow its actual catalog schema.
- Official `zaic_tools` will replay the submitted calls. Do not write custom app backends or modify official executor/validator to influence scoring.

## Workflow
1. **Phase 0 (required):** Inspect the complete local SDK: `README.md`, `public_test/zaic_tools/__init__.py`, `executor.py`, `validate.py`, `apps/base.py`, system/meta handler, tool catalog, and 1–2 input examples. The provided context excerpt is truncated, so do not assume the full code is known. Record verified behavior and unknowns in `docs/official-contract.md`.
2. Implement a **text-only**, end-to-end tool-use baseline first. Build the real audio path later. Use a single model/tool loop with strict validation; do not begin with multi-agent or LangGraph.
3. Keep runtime code independent of challenge file layout through a thin adapter, but do not build unnecessary abstraction layers. Track explicit tool-call budget, termination reason, steps, model usage, and execution observations.
4. Add local diagnostics and submission verification; do not claim an official success score without an actual official scorer or ground truth.
5. Integrate ASR for the final audio **and history user audio**, then optimize from measured error categories.

## Source edits and test discipline
- Keep changes narrow, readable, fully typed where reasonable, and testable without GPUs or network.
- Use pathlib, Unicode UTF-8, and cross-platform APIs. PowerShell instructions for Windows; avoid assuming Bash.
- Use fixtures/mocks for unit tests; use the real SDK for integration tests. Do not modify official dataset or SDK. Never silently substitute mocks during real submission.
- No repeated state-changing tool call after a timeout unless it is confirmed safe; avoid duplicate writes/transfers/orders.
- Never treat invalid schema, `ok: false`, or no-op as success. Handle errors explicitly.
- Tests must cover installed-app restrictions, `x-one-of`/type requirements as implemented by SDK, read-before-write, dependent actions, history, clarification, out-of-scope, no-op, call order, and state hash.
- Keep `.env` and any credentials out of git. Do not require paid external APIs in the test suite; use stubs.
- Prefer CLI and simple modules. No FastAPI, Redis, databases, queues, background workers, dashboards or Docker until a verified requirement demands them.

## Reporting after each phase
Report: (1) verified changes/files, (2) exact commands run and results, (3) SDK contract findings, (4) remaining blockers, (5) next step. Never claim tests passed if not executed. Update `PLANS.md` and `harness/build-log.md` with evidence. Do not push, deploy or submit externally without user approval.
