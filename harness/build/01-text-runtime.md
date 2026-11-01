# Phase 01 — Text-only runtime baseline

## Deliverables
- Python project configuration (`pyproject.toml`, optional uv lock, tests).
- Read-only JSONL loader, context and allowed-tool catalog lookup.
- A thin real SDK adapter with one `ToolEnv` per sample inference.
- CLI for selecting a public-test sample, supplying an explicit transcript **for local debugging**; text override is not a substitute for audio in final public-test batch.
- Working ordered call execution via `env.validate` then `env.call`, preserving observations, `env.get_tool_calls()` and `env.get_state_hash()`.

## Acceptance
- One official example sample can initialize one ToolEnv and generate a structurally valid result, with an intentional mocked/deterministic decision source only for testing.
- Tests prove calls are never sent when tool not permitted or schema invalid.
- Tests prove invalid attempts made through SDK remain in recorded trace (no filtering), and previously executed history actions are not replayed.
- No direct raw state access in agent. No train access.
- Tests run locally without external model/network/GPU. Optional integration sample uses exact official kit.
