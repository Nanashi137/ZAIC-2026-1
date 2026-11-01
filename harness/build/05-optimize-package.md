# Phase 05 — Optimize and prepare reproducible package

Dependency: Phase 04.5 compliance controls must remain enabled. The final organizer-facing entrypoint must use the
per-sample outer watchdog and a fully offline local LLM backend.

## Deliverables
- Versioned experiment configurations and compare runs on a fixed development subset; track tokens, wall time, valid call rate, no-ops, and error categories. Only use official score when available.
- Optimize likely errors: poor transcription, prompt verbosity, tool routing, `x-ref` IDs, clarification, context/history representation and premature stopping.
- Pin dependencies and write Windows setup + run instructions for a fresh machine.
- Inspect **new official documentation** before implementing private-test Docker packaging. Do not invent private submission requirements or assume external LLM APIs are allowed.

## Acceptance
- A clean local environment can reproduce an example and batch submission when dependencies/configuration are supplied.
- Tests and schema validation pass; experiment comparisons show measured evidence rather than narrative claims.
- Private-test packaging, if required, follows official rules and is validated separately.
