# Official toolkit contract — public-test only

**Verified locally:** 2026-10-30 against toolkit release `v3.2.0` (built
`2026-10-24T23:35:25+0700`), catalog schema `2.7.0`. The installed files under
`data/train+public_test/public_test/zaic_tools/` are authoritative. No `train/`, audio contents, or raw
`state.json` contents were inspected.

## Kit and input contract

- The dataset root is the directory containing `public_test/`; here it is `data/train+public_test/`.
- `public_test.jsonl` has 634 unique rows with exactly `id`, `audio`, `context`, and `history`. Public test has no
  reference transcript or label.
- `context` contains `current_time`, `timezone`, `location`, `user_profile`, and `installed_apps`. Each installed-app
  entry names the sample-specific tools that may be called.
- `audio` identifies the final user request. A user history turn may identify another audio file. Assistant history
  turns can include `text`, `tool_calls`, and `tool_results`.
- Dataset-root `README.md`, section 2.1, explicitly says history actions are already reflected in the initial
  environment. They provide conversational context only and must never be replayed.
- Inspection found 419 samples with history and 1,299 user-history audio references. These are structural counts,
  not transcripts or labels.

Source: dataset-root `README.md` sections 2.1–2.3 and two inspected input shapes (`public_0002`, `public_0003`), plus
one history-bearing shape (`public_0006`).

## Catalog and per-sample permissions

- The catalog contains 36 apps and 122 tools: 52 `read`, 69 `write`, and one `meta`. All 122 are implemented by the
  installed SDK (`python -m zaic_tools ... --apps` reported 0 missing).
- `Simulator.__init__` forms `allowed` from installed tool names and unconditionally unions
  `{"sys.ask_missing_info"}` (`executor.py:35-45`). `Session.__init__` derives installed tools from the environment
  `_meta.tools` list (`executor.py:79-93`).
- The runtime must offer the intersection of the catalog and the public input's installed tool names, plus the
  verified system tool. Phase 01 also compares this public-input set with `Session.allowed` and fails closed on a
  mismatch.
- A catalog-known but non-installed tool is a schema rejection. The SDK does not infer permission from the app name;
  it checks the full tool name.
- The system catalog description says `sys.ask_missing_info` is always available even when absent from
  `installed_apps`. Its exact schema requires:

  ```json
  {"tool_name": "<app.tool>", "params": ["<missing_parameter>"]}
  ```

  Both fields are required. Execution only records the call, changes no state, and returns
  `{"ok": true, "waiting_for_user": true}` (`apps/sys.py:7-13`).

## Official API and lifecycle

Verified public factory (`zaic_tools/__init__.py:32-41`):

```python
ToolEnv(split: str, question_id: str, root: str | Path | None = None) -> Session
```

It opens `<root>/<split>/<split>_environment/<question_id>/state.json` and the split catalog, then returns:

```python
Session(sample_id: str, entry: dict, catalog: dict | None)
```

Relevant lower-level API:

```python
Executor(env_path: str, catalog_path: str | None = None)
Executor.open(sample_id: str) -> Session
Executor.replay(sample_id: str, tool_calls: list[dict]) -> tuple[list[dict], dict]

Session.validate(name: str, arguments: dict | None = None) -> dict
Session.call(name: str, arguments: dict | None = None) -> dict
Session.get_tool_calls() -> list[dict]
Session.get_state_hash() -> str
```

Lifecycle rules:

1. Construct exactly one `ToolEnv` for one sample inference. A second construction reloads pristine state and is not
   a retry mechanism.
2. Call `validate` before `call`. Validation neither executes nor records a call (`executor.py:64-67`).
3. `Session.call` copies `{name, arguments}` into its ordered call list before execution/validation. Consequently, a
   call rejected by the SDK remains in `get_tool_calls()` (`executor.py:95-103`).
4. App state is copied into memory. The official source `state.json` is not modified.
5. Build submission fields only with `get_tool_calls()` and `get_state_hash()`. Do not reconstruct or filter them.

`get_state()` exists for scorer/train debugging, but the competition agent must not call it or read raw `state.json`
as planning input. The Phase 01 adapter intentionally omits it from its runtime protocol.

## Authoritative schema behavior

`env.validate`, execution, CLI validation, and replay all use `validate.schema_error` (`validate.py:100-122`). It is a
small official validator, not general JSON Schema.

It rejects:

- a tool absent from the catalog or sample allowlist;
- a required value that is `None`, `""`, `[]`, or `{}`;
- wrong declared types (booleans are not integers/numbers);
- enum mismatches, including exact boolean type matching;
- non-finite numbers and numeric range violations;
- invalid `date`, `date-time`, or `time` formats as implemented in `format_ok`;
- array items violating their declared `type` or `enum` recursively;
- parameter-level `x-one-of` when no complete group is present.

Five catalog tools use `x-one-of`: `contacts.update`, `phone.call`, `sms.send`, `ewallet.transfer`, and `music.play`.
The rule is “at least one entire group is non-empty”; it does not enforce exactly one group.

Important deviations from generic JSON Schema expectations:

- Undeclared/additional argument keys are not rejected by `schema_error`.
- Empty optional values are skipped rather than type-checked.
- `x-ref` (61 occurrences) and `x-match` (38 occurrences) are not read by the SDK validator. They are catalog guidance
  for planning/scoring: obtain referenced IDs through permitted reads and preserve fuzzy free text. They must not be
  presented as locally validated referential integrity.
- A catalog-valid but unimplemented tool would pass `validate` and be rejected by `call`; this release implements all
  122 catalog tools, so that discrepancy is presently theoretical.

## Execution results, failures, and no-ops

- A schema rejection returns `{"ok": false, "error": ...}`, changes no state, and is recorded only if it reached
  `call`.
- App-level conditions such as an unknown ID, insufficient balance, or an already-completed operation are normally
  not schema errors. `App.noop` returns `{"ok": true, "done": false, "message": ...}` and leaves state unchanged
  (`apps/base.py:32-35`).
- Reads with no matches usually return `{"ok": true, "<collection>": []}`.
- `Simulator.call` catches an unexpected app exception and returns the same no-effect shape with a message prefixed
  by the exception class, for example `ValueError: ...`; it does not raise into the agent loop (`executor.py:48-62`).
- Therefore `ok: true` alone is not proof of a requested state change. Phase 01 distinguishes SDK rejection,
  executor exception, explicit `done: false` no-op, read execution, write execution, and meta execution. “Write
  succeeded” reflects the SDK's normal result convention, not hidden semantic ground truth.
- Never blindly repeat a state-changing call after an ambiguous timeout or observation failure.

## State hash

`Session.get_state_hash()` delegates to `utils.state_hash` (`executor.py:109-111`, `utils.py:49-67`):

1. `canon` removes `_meta`, the read-only live apps `fuel_price`, `gold_price`, and `weather`, and recursively removes
   free-text fields listed in `FREE_TEXT_KEYS`.
2. It normalizes string case/whitespace and number-unit spacing; lists of scalars are sorted, while lists containing
   dicts retain order.
3. It serializes canonical state as sorted, compact ASCII JSON and returns SHA-256.

A valid or unchanged hash does not establish task success. Some read-only, meta, no-op, live-app, or free-text-only
differences cannot be interpreted as a public score from the hash alone.

## Submission and validator limits

Each UTF-8 JSONL row is:

```json
{"id":"public_XXXX","tool_calls":[],"state_hash":"<SDK value>"}
```

`tool_calls=[]` is valid for out-of-scope requests. The installed CLI validator opens each row's environment and runs
the official schema validator on its calls. Source inspection of `zaic_tools/__main__.py:13-29` confirms it does **not**
check full split coverage, duplicate IDs, row-key exactness, state-hash correctness, intent, or semantic success.
Those need separate structural checks and, where available, official evaluation.

## Still unknown

- Exact official scoring weights/details beyond the SDK README's replay/state comparison and tool-call F1 summary.
- Public hidden labels and semantic task success for any public-test sample.
- Final private-test Docker protocol and whether external network/model services are allowed.

These remain blockers for later optimization/packaging, not for the completed text-only baseline.
