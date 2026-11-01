# Project goals and non-goals

## Primary outcome
Generate a competition-valid `submission.jsonl` for the **public_test** split using the provided `zaic_tools` runtime, with a local inference path:

`audio / optional text override → context + history → model → schema validation → actual tool calls + observations → env.get_tool_calls() + env.get_state_hash()`.

## Success gates
1. The project can inspect the official kit without modifying it.
2. A text-only sample runs through the real `ToolEnv` and produces the official output shape.
3. The agent can perform dependent read→write actions and use observation data for IDs/parameters.
4. Clarification, out-of-scope, invalid/failed tool, and no-op paths work.
5. Audio mode handles both final request and audio-backed user history turns.
6. Batch mode emits exactly one result per input ID and passes independent format checks and official schema validator.
7. Local metrics are reported accurately; never conflate state-hash consistency with task success.

## Non-goals, for now
- Any access to `train/`, training/fine-tuning, or train labels.
- Building fake banking, SMS, calendar or home-automation services.
- Reading source `state.json` as an information shortcut for the agent.
- Heavy multi-agent frameworks, product UI, database, backend server, distributed architecture.
- Blind optimizing against public leaderboard without disciplined experiments.

## Operating constraints
SDK contracts and evaluation rules take priority over this plan. Private-test Docker requirements remain unknown until official instructions are available. Avoid data leakage and always keep models/providers configurable.
