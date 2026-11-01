# LifeMate Codex starter pack (instructions + inspection helpers)

This ZIP is intended to be **extracted into the root of your own LifeMate code repository**. It supplies Codex instructions, planning/evaluation documents, phase briefs, and two **read-only / structure-only** validation scripts. It deliberately does **not** contain your private Zalo data, the official SDK, or the agent implementation.

## Start on Windows 11
1. Place this pack at the root of your code repository. Keep the original competition kit at `data/train+public_test/` (or edit the configured path in prompts).
2. Open that repository folder in VS Code with Codex. `AGENTS.md` at the root is discovered by Codex automatically.
3. Run `python .\scripts\inspect_kit.py --root '.\data\train+public_test'`.
4. Open `PROMPTS.md`, paste the **First prompt** into Codex, and let it inspect the local full SDK and implement Phase 01.
5. After each phase, follow the **Subsequent prompt** with the next phase file.

## Pack contents
- `AGENTS.md` — durable instructions and non-negotiable rules for Codex.
- `GOALS.md`, `PLANS.md` — outcomes, phase gating and progress.
- `PROMPTS.md` — pasteable prompts.
- `docs/official-contract.md` — confirmed information from provided excerpt and verification checklist.
- `docs/architecture.md`, `docs/evaluation.md`, `docs/windows-setup.md` — design, diagnostics, and Windows commands.
- `harness/build/` — per-phase coding-agent brief and acceptance criteria.
- `harness/build-log.md` — progress/test evidence record.
- `scripts/inspect_kit.py`, `scripts/verify_submission.py` — dependency-free local helpers, not official evaluation.

**Source note:** Toolkit extract supplied in chat, v3.2.0, is partial. Codex must inspect the full local official SDK before relying on detailed behavior. No code in this pack accesses `train/`.
