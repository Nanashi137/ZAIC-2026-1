# Phase 02 — LLM-driven sequential decision loop

## Deliverables
- Configurable OpenAI-compatible model client; deterministic fake implementation for tests.
- Context builder supplying last-user transcript, history, current_time/timezone/profile, and **only permitted tool definitions**.
- Agent loop: propose next action / finish; local permission check; official `env.validate`; execute; append actual observation; repeat.
- Bounded steps, retry on invalid *proposals* without executing them, explicit handling of model errors, no-op (`done:false`), missing-info tool and unsupported request.
- Logs of decision, validated action, actual observation and termination reason (scrub sensitive details before external export).

## Acceptance
- Prove a dependent read→write sequence uses a returned identifier, not guessed state.
- Prove installed-app restrictions, sys/meta semantics verified in Phase 0, and `x-one-of` where required.
- Prove no repeated write after uncertain execution and no reset within inference.
- Tests for missing info, tool-unavailable, no-op, failures, model refusal/bad output, max-step cap.
- Real SDK integration smoke test; exact commands and outcomes logged.

## Completion evidence — 2026-10-30

- Implemented provider/context/tool-schema/model-loop boundaries in `lifemate/model.py` and `lifemate/agent.py`.
- Extended the shared runtime with model failure termination, invalid-proposal retry, duplicate-write blocking, and
  stop-on-uncertain-execution behavior.
- `uv run pytest -q` passed 25 tests, including an observation-derived read→write dependency and a real-SDK scripted
  model smoke.
- `uv run ruff check lifemate tests scripts` passed.
- `python -m scripts.phase02_smoke` produced an SDK-derived public-test row; official validation reported one line and
  zero schema errors.
- No external model API, audio, raw state, `train/`, deployment, or submission was used.
