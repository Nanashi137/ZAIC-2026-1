
You are the lead AI engineer implementing a competition-compliant solution for Zalo AI Challenge 2026 — LifeMate.

WORKSPACE AND INSTRUCTIONS
- Current environment: Windows 11, VS Code, PowerShell. Use Windows-compatible commands, pathlib, and UTF-8.
- Repository root: the current workspace.
- Official dataset root: data/train+public_test/.
- Work only with public_test/ and the dataset-root README.md and VERSION. Do not inspect, index, copy, modify, or train on train/.
- First read AGENTS.md, GOALS.md, PLANS.md, docs/official-contract.md, docs/architecture.md, harness/build/00-sdk-discovery.md, harness/build/01-text-runtime.md, and harness/build-log.md.
- The installed official kit and SDK source are authoritative. Update project documentation when an earlier assumption contradicts the real kit. Do not modify the official kit.

ALREADY VERIFIED ON THIS MACHINE
- Kit release: v3.2.0, built 2026-10-24T23:35:25+0700.
- Catalog schema_version: 2.7.0.
- Public test: 634 unique samples; all 634 audio paths and state.json files exist; 0 duplicate IDs.
- 419 samples include conversation history, with 1,299 references to user history audio.
- Catalog: 36 apps, 122 tools; 52 read / 69 write / 1 meta.
- Schema extensions present: x-ref, x-match, x-one-of.
- Official SDK check: 36 apps, 122/122 tools implemented, 0 missing.
- Inspection already ran successfully:
  python .\scripts\inspect_kit.py --root '.\data\train+public_test'
  Push-Location '.\data\train+public_test\public_test'
  python -m zaic_tools --split public_test --root .. --apps
  Pop-Location

These checks confirm kit structure and implementation coverage, NOT task success or an official evaluation score. You may rerun them to verify your environment.

YOUR CURRENT MISSION: COMPLETE PHASE 00 AND PHASE 01 ONLY

Do not jump to ASR, model optimization, batch submissions, a frontend, LangGraph, or a multi-agent system.

Phase 01 needs a functioning text-only SDK baseline, not a smart production agent. The later LLM-driven tool loop belongs to Phase 02.

PHASE 00 — DISCOVER AND DOCUMENT THE REAL CONTRACT

1. Inspect the full, local public_test/zaic_tools source, especially __init__.py, executor.py, validate.py, apps/base.py, the sys/meta tool implementation, and any helpers needed to understand behavior. Examine the catalog, public_test JSONL structure, and sample submission. The provided excerpts may be truncated; inspect the original files.

2. Verify exact ToolEnv/Session signatures, allowed-tools rules, schema validation semantics, execution results and recording, exceptions, no-op behavior, and hash behavior.

3. Verify how the sys.ask_missing_info tool is offered/permitted and its arguments/observations. Do not guess.

4. Inspect x-one-of, x-ref, and x-match behavior. Use the official SDK validator as authority; do not substitute a generic JSON Schema validator without checking compatibility.

5. Confirm that history tool effects are already reflected in the initial environment and must NOT be replayed.

6. Update docs/official-contract.md and docs/architecture.md with confirmed behaviors, file/method references, unresolved questions, and minimal component boundaries. Record deviations from existing documents.

7. Do not use state.json or env.get_state() to supply the agent with privileged app state. It is permissible for the official ToolEnv to load its state internally. Any offline diagnostics must be explicitly separated from agent runtime.

PHASE 01 — IMPLEMENT A REAL TEXT-ONLY RUNTIME

1. Inspect existing repository code before modifying it; preserve useful code. Create the smallest maintainable Python package/CLI and tests necessary. Prefer Python 3.11+, uv if installed, pytest, and standard library where feasible. A deterministic decision source or stub is acceptable for Phase 01 integration testing; make it explicit and do not present it as an LLM solution.

2. Implement a read-only loader for public_test.jsonl and sample-specific context/history/installed_apps, plus a catalog lookup restricted to tools actually allowed for the current sample. Respect any verified SDK-specific meta-tool rules.

3. Implement a thin adapter around the official ToolEnv. In each sample inference, instantiate exactly one ToolEnv('public_test', sample_id, root=dataset_root). Never reset or reinstantiate it to revise an action sequence. Do not reimplement Zalo's 122 tools or mutate its SDK.

4. Provide a CLI command that takes a sample ID and an explicit text transcript (public-test has no reference transcript). The transcript override is for debugging only and must NOT be used as if it were the sample's ground-truth audio.

5. Implement action preflight using env.validate(name, arguments), check sample-specific permissions, then call env.call(name, arguments) only for validated actions. Capture real observations. Distinguish rejected calls, tool errors, and ok:true/done:false no-ops from successful effects. Failed attempts actually passed to env.call must remain in the official call trace.

6. Generate the submission record strictly from the SDK:
   {
     'id': sample_id,
     'tool_calls': env.get_tool_calls(),
     'state_hash': env.get_state_hash()
   }

   Never fabricate a hash, reconstruct the call list, or silently drop failed/rejected calls.

7. Ensure previous history actions are never replayed. Keep read-before-write and dependency handling testable, while leaving the LLM-driven planning loop to Phase 02.

8. Implement focused unit tests (no network/GPU required) and integration/smoke tests using the actual public-test SDK. A controlled deterministic tool selection for testing is fine, but it must obey permissions and schema and use an actual ToolEnv. Use a new session only for an independent test case, never to undo a decision within one inference.

9. Demonstrate one real sample inference producing a well-formed id/tool_calls/state_hash record. Verify submitted calls with the official SDK validation command where applicable; clearly distinguish structural/schema validity from semantic success.

NON-NEGOTIABLE CONSTRAINTS

- Do not read or alter train/; do not commit, share, or upload any official dataset files/audio/state or SDK materials.
- Do not expose raw state.json or env.get_state() to the agent as runtime context.
- Do not assume all 122 tools are permitted for a specific sample; select from installed_apps and verified system-tool rules.
- Do not generate, guess, or hardcode tool arguments or IDs based on sample IDs.
- No reset/second ToolEnv within the same sample inference to search for a favorable result.
- Do not retry state-changing calls blindly; a timeout/observation failure does not prove no write occurred.
- Do not send private data to external LLM APIs without my approval and a check against challenge rules. Tests must run offline.
- Do not claim an official score from a valid state_hash or passing schema validation.
- Do not push, deploy, or submit externally without my approval.

COMPLETION AND REPORTING

- Implement code, not just a design document. Independently run tests and fix failures.
- Update PLANS.md and harness/build-log.md with exact commands, pass/fail outcomes, files changed, and remaining issues.
- Mark Phase 00 and Phase 01 complete only after satisfying their individual acceptance criteria.
- Finish with a concise report containing:
  (1) SDK findings and deviations
  (2) implemented files/interfaces
  (3) commands/tests actually run and outputs
  (4) one sanitized sample execution trace/submission record
  (5) blockers
  (6) precise next step for Phase 02
- If any required files or tools are inaccessible, clearly describe the blocker and finish everything possible without inventing official behavior.

Start by reading the repository instructions and inspecting the SDK. Then proceed through Phase 00 and Phase 01 without asking for routine confirmations.
