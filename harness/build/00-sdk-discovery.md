# Phase 00 — Official SDK discovery

## Scope
Inspect **only** `data/train+public_test/public_test/` and the dataset-root `README.md`/`VERSION` (metadata only). First run `scripts/inspect_kit.py`. Read the full official `zaic_tools` API, validator, executor and system/meta tool. Review catalog extensions, sample input and example submission; do not read entire bulk data or any raw state.json as agent context.

## Deliverables
- Correct/extend `docs/official-contract.md` with actual class/function signatures, rules and unresolved questions; reference file paths and definitions.
- Propose minimal module boundaries in `docs/architecture.md`.
- Document exact input/audio/history semantics, sample-specific permissions, system tool access, output schema, execution/no-op/validation behavior.

## Acceptance & verification
- `python scripts/inspect_kit.py --root 'data/train+public_test'` succeeds.
- Official `--apps` reports implemented catalog tools (record missing/unsupported tools, if any).
- The full SDK, not the truncated extract, supports every claimed API behavior.
- No official file changed; no train access; no model/network dependency required.
- Report every uncertainty and close only those verified by source.
