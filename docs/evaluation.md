# Evaluation and debugging protocol

## Phase 04.6 — labeled challenge-score estimate (2026-11-02)

The existing semantic evaluator had positional tool/argument diagnostics and gold-call replay hash checking,
but no challenge formula. The new primary metric is **`challenge_score_estimate`**, never an official-equivalent
`challenge_score`. `challenge_score_exact` is always false until the organizer's exact call matcher is available.

### Evidence and matching limits

- The split SDK README describes replay, state comparison and call F1, but ships no call scorer/matcher.
- Dataset README section 2.2 describes `x-match: fuzzy`; SDK `utils.py` comments also say free-text arguments are
	scored fuzzily. Neither provides that algorithm, its thresholds, partial-match rules, ordering rules or
	one-to-one assignment rules. Search helpers in app implementations are **not** scoring equivalence functions.
- `executor.Session.call` records a call before schema validation; those rejected attempts remain submitted calls.
	Local proposals rejected before SDK dispatch never enter the actual submission and are diagnostic only.
- Fallback matching uses unordered multiset intersection, with one-to-one consumption of gold calls. Names and
	all argument fields must be exact canonical JSON. Object key order is irrelevant; array order, JSON numeric
	representations/types, booleans, case and whitespace remain significant. No fuzzy, partial or LLM matching.
	Reordering otherwise identical calls does not affect M, but actual execution order still affects state/hash.
- SDK-rejected submitted calls are excluded from matching and counted in U. Duplicate submitted calls consume
	only available gold multiplicity; excess duplicates enter U. Matching never rewrites submitted traces/hashes.
- Dataset README section 2.1 defines `label.final_state` as **changed apps only**, not the complete final state.
	Hashing it directly against the submitted full-state hash is invalid. The initial implementation did so;
	its 100-sample reports with zero hash matches must not be used to compare challenge scores.
- Gold full-state hashes now come from isolated **selected-split SDK reference sessions**, replaying only the
	labeled current-turn calls with `validate` then `call`, then taking `get_state_hash()` unchanged. Gold replay
	runs after the mode's entire inference pass, outside all sample inference budgets, and is cached by dataset,
	split and sample ID across A/B modes. It never uses an inference session or retries an inference plan.
	No direct environment-file access, `get_state()`, label-state merging or gold input to planning is used.
- Reference replay errors are serialized as unavailable hashes and `reference_state_hash_error`, not disguised
	as hash mismatches or equal states. `label.final_state` is not used as a full-state substitute, even when empty.
- Constrained JSON receives the same adapted argument schemas/descriptions as native tool calling, including
	`agent.finish`'s required status/reason. Its `oneOf` output schema binds each allowed official action name to
	that action's parameters; optional fields remain optional. Mode-specific instructions require a JSON next action
	instead of the native no-JSON completion instruction. SDK preflight remains authoritative for cross-field rules.

### Formula and failure policy

The formula is supplied by the user's official-scoring specification; exact call matching remains unverified.
For usable predictions: empty G scores 1 iff S is empty, without a hash requirement. Otherwise a hash mismatch
or M=0 scores 0; matching hashes and M>0 score F1, with precision M/(M+U) and recall M/len(G). Zero denominators
are handled as zero. Missing gold/system hashes for nonempty G are reported explicitly and score zero, not
presumed equal. Unusable predictions score zero even with empty G. An available genuine SDK trace/hash is
scored as-is even when inference terminated with an error or timeout; termination remains a diagnostic.

Aggregate estimate = 100 × sum(sample estimates) / **all evaluated samples**, including failures and zeroes.
Empty selection produces 0 with total_samples=0. Existing successful-run diagnostic denominators are unchanged.
No challenge metric is produced for unlabeled public test by the train-only `evaluate` command.

### Outputs

Existing output filenames and fields are retained. Added top-level per-record fields include:

```json
{
	"challenge_score_estimate": 0.8,
	"challenge_score_exact": false,
	"matching_method": "strict-canonical-json-unordered-one-to-one-v1",
	"hash_match": true,
	"gold_calls": 2,
	"submitted_calls": 3,
	"matched_calls": 2,
	"unmatched_calls": 1,
	"schema_rejected_submitted_calls": 0,
	"precision": 0.6666666666666666,
	"recall": 1.0,
	"f1": 0.8,
	"score_reason": "matched_f1"
}
```

This is a synthetic illustration, not a measured model result. Aggregate fields include
`challenge_score_estimate` (0–100), `challenge_score_exact`, `matching_method`, `total_samples`,
`mean_sample_score`, perfect/partial/zero counts and rates, hash match/mismatch/unavailable counts,
`hash_match_rate` over all samples, total gold/submitted/matched/unmatched/rejected calls, `score_reasons`,
`zero_due_to_hash_mismatch`, `zero_due_to_no_matched_calls`, and empty-gold special-case counts.
The two zero-score gate counts can overlap. Unknown hashes are reported separately, not called mismatches.
Worst examples rank by sample challenge estimate; positional/relaxed errors remain diagnostic, not scoring rules.

New per-record reference provenance: `reference_state_hash_source="sdk-gold-replay-full-state-v1"`,
`reference_state_hash_error` (null on success), and `sdk_rejected_call_indices`. Model-request diagnostics
are retained; all nested read-only diagnostic mappings are recursively converted to JSON-native values before export.
Regression coverage uses the real runtime and scripted LLM decisions in both modes with raw capture on/off.
Raw responses are still opt-in. The legacy `successful_samples` field means **usable result rows**,
not semantic success, and `failure_rate` still means missing results for backward compatibility. Reports and
comparisons now distinguish:

- `infrastructure_failure_samples/rate`: no usable evaluation result, including setup/ASR exceptions.
- `abnormal_termination_samples/rate`: usable results whose controller did not finish normally (such as budget,
	malformed-output-limit, duplicate-recovery or timeout termination). Missing-result failures are separate.
- `malformed_output_proposals`, `malformed_output_samples`, `invalid_output_rate`: malformed-output steps and
	all-sample rate of records with at least one such step; this is no longer an alias for missing-result failures.
- `schema_rejected_proposals`, `schema_rejection_samples`, `schema_rejection_sample_rate`: local preflight errors,
	not submitted SDK rejections. `invalid_finish_proposals/samples` and `invalid_finish_sample_rate` are separate.
- `timeout_count/rate`: actual timeout sample count divided by all evaluated samples, not a count labeled a rate.
- `reference_hash_failure_samples`, `termination_reasons`, `step_status_counts`: reference and controller details.

All new error rates use all evaluated samples; categories may overlap and are diagnostic rather than scoring rules.

A/B comparison puts the estimate first and includes hash/perfect/partial rates plus existing diagnostics.
It verifies unique identical ordered sample IDs and reports `improved_zero_to_nonzero`,
`regressed_nonzero_to_zero`, `became_perfect` and `lost_correct_state_hash` IDs. Native/constrained columns
remain correct regardless of which mode was run first. Every inference call creates its own SDK session;
each gold reference uses a separate offline SDK session, cached across modes. Existing cooperative 28-second
inference deadlines are propagated to both the runtime and model decision source; outer watchdog requirements remain.
Reference replay is evaluation work, not timed inference.
The pre-gold-replay 100-sample artifacts with invalid partial-state hashing remain historical only; never mix them
with the reviewed `eval-score-ab-100-new` and `eval-score-ab-100-asr` outputs or a future run directory.

Aggregate output now includes `run_identity` (dataset path/split/catalog version, selected-ID hash, model, endpoint,
output/input mode, seed, budgets, timeout and ASR configuration), `mean_model_latency_seconds`,
`p95_model_latency_seconds`, ASR latency totals, turns, cache hits and
cache-hit rate, plus input/output/total tokens and tokens per usable sample. Sample wall latency is retained but
explicitly labeled ASR-cache-sensitive. In a sequential A/B run,
the first mode can populate the shared content-addressed ASR cache and the second can reuse it; therefore model-only
latency is the fair output-mode timing comparison. Semantic comparison remains paired because both modes receive the
same transcript content.

Constrained JSON requests use `response_format` without also sending empty native `tools` or native `tool_choice`
controls. Both modes enforce the same exact `agent.finish` contract: `status` and a non-empty `reason` are required;
an empty object is invalid rather than silently converted to success.

### Reviewed 100-sample A/B evidence (seed 42)

| Input | Mode | Estimate | Hash match | Perfect | Partial | Tool name | Argument | Model requests/sample |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| oracle | native | 35.3960 | 0.57 | 0.21 | 0.24 | 0.6098 | 0.4773 | 4.21 |
| oracle | constrained JSON | 22.9167 | 0.36 | 0.19 | 0.06 | 0.3933 | 0.3143 | 3.13 |
| ASR | native | 12.9359 | 0.30 | 0.08 | 0.09 | 0.4045 | 0.2414 | 4.26 |
| ASR | constrained JSON | 10.7333 | 0.27 | 0.08 | 0.05 | 0.3100 | 0.2146 | 2.98 |

Native wins the primary estimate by 12.4793 points with oracle text and 2.2026 with ASR, so constrained JSON is not
the selected production mode. It reduces requests and malformed native-output risk, but on this sample it loses more
semantic coverage than it gains in format control. Oracle-to-ASR degradation is 22.4601 points for native and
12.1833 for constrained JSON, making transcription/utterance understanding the largest observed bottleneck.

The saved ASR wall-time delta is invalid as a mode-speed claim: native had 0/333 cache hits and incurred 228.16 s of
ASR inference, while constrained JSON reused 333/333 turns. Those artifacts also predate `run_identity`, so their ASR
checkpoint cannot be proven from the files. Future runs default to `Qwen/Qwen3-ASR-0.6B` and record the exact setting.
These are partial train results, not official-equivalent scores or reliable public/private score predictors.

### Phase 05 promotion protocol

1. Keep native tool calling as the control. Constrained JSON remains a diagnostic candidate until it improves the
   primary estimate on every held-out fold rather than only reducing requests or malformed responses.
2. Create several fixed, stratified train folds using only generic metadata such as app/tool family, gold call count,
   history presence and read→write dependency depth. Tune on some folds and require gains on untouched folds; never
   encode sample IDs, transcript phrases or public-test patterns into prompts/controller logic.
3. Run oracle and cached-ASR inference on identical IDs. Attribute paired degradation first to transcription versus
   planning, then report confidence intervals across folds rather than optimizing a single 100-sample mean.
4. Prioritize generic, catalog-derived improvements: preserve required arguments, ground `x-ref` identifiers through
   observations, stop unnecessary reads once dependencies are resolved, and avoid extra writes. The reviewed native
   oracle run submitted 317 calls for 205 gold calls and matched 114; ASR matched only 66, so both precision and
   utterance understanding need work.
5. A/B serving independently from controller changes: current Q4 GGUF versus a higher-precision 4B quantization or
   BF16 engine, prompt/context caching, and provider-native Qwen tool parsing where supported. Quantization/parser
   choices do not change parameter count, but the combined planner+ASR checkpoint must remain within the 6B rule.
6. Promote a configuration only after schema/coverage checks, stable runtime under the 28-second budget, improvement
   across held-out folds and no regression in empty-gold/unsupported/clarification safety. Use the public leaderboard
   only as a sparse final validation signal, never as a pattern-mining set.

### Local commands (run from project root, Bash)

Ten-sample native evaluation:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
uv run python -m lifemate.cli evaluate \
	--base-url http://127.0.0.1:1990/v1 \
	--model qwen3.5-4b \
	--output-dir outputs/eval-score-native-10 \
	--agent-output-mode native \
	--input-mode oracle \
	--limit 10 \
	--seed 42 \
	--sample-timeout-seconds 28 \
	--competition-mode
```

Ten-sample paired A/B evaluation:

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost \
uv run python -m lifemate.cli evaluate \
	--base-url http://127.0.0.1:1990/v1 \
	--model qwen3.5-4b \
	--output-dir outputs/eval-score-ab-10 \
	--agent-output-mode native \
	--compare-with-mode constrained-json \
	--input-mode oracle \
	--limit 10 \
	--seed 42 \
	--sample-timeout-seconds 28 \
	--competition-mode
```

Per-mode artifacts: `evaluation-records.jsonl`, `evaluation-metrics.json`, `evaluation-report.md`.
Paired artifacts: `comparison.json`, `comparison.md`. These commands are supplied for manual runs; no full
training evaluation is part of implementation verification.

## What the public kit proves (and does not)
- `env.validate` / official `--validate`: validates tool-call schemas and availability; **does not score intent correctness**.
- `env.get_state_hash`: attests to local simulator state resulting from actions; **does not prove target state was reached**.
- `public_test.jsonl`: no labels or transcripts. It cannot directly yield ground-truth tool accuracy, task success, WER or CER.
- Do not invent benchmark scores or describe a local heuristic as an official score.

## Required reproducible checks
1. Data sanity: every input `id` unique; path exists; sample count obtained from input, not hardcoded.
2. One sample: real ToolEnv initialization, allowed tool filtering, `validate` then `call`, ordered tool-call retrieval, SDK hash retrieval.
3. Batch: exactly one output row per ID; no duplicates, missing or extra IDs; valid UTF-8 JSONL; `tool_calls` array with `{name,arguments}`; opaque `state_hash` non-empty string.
4. Run official `zaic_tools --validate` against output. Its success is **schema-only**.
5. Optional isolated replay/check: create a *separate verification session* after inference and replay recorded calls, compare the resulting hash. Never feed this verification session back into the inference agent; it is not a retry.
6. If real official scorer/leaderboard provides task score, record model, prompt, code revision, dataset version, timestamp and score. No private labels leakage.

## Error taxonomy
- Audio decode / ASR / history audio transcription
- Context extraction (time zone, relative time, profile)
- Intent or multi-intent decomposition
- Tool availability or wrong app selection
- Schema, `x-one-of`, enum, type, required fields
- `x-ref` grounding (IDs), fuzzy text matching
- Read-before-write and dependency/order
- No-op (`ok:true,done:false`), tool failure, transient model error
- Missing-info (`sys.ask_missing_info`) / unsupported request
- Unexpected extra writes / repeated irreversible side effects
- Submission ID/shape/count/hash consistency

## Experiment record
Per experiment capture: `{git_revision, dataset_version, split, sample_ids, model, asr_model, settings, max_steps, prompt_version, tool_validity_rate, noop_rate, failures_by_category, runtime, total_tokens_if_known, official_score_if_any}`.

Keep traces local. Summarize aggregates by default; do not commit user profile details or raw audio/transcripts to source control. Code should work with a fake deterministic model and no network for unit tests.
