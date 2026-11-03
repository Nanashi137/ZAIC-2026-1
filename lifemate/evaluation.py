"""Semantic evaluation on labeled splits for controlled agent experiments."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from tqdm.auto import tqdm

from .audio_pipeline import AudioInferencePipeline
from .compliance import SampleDeadline, SampleDeadlineExceeded
from .data import DatasetError, DatasetReader, Sample
from .runtime import DecisionSource, RunResult
from .scoring import REFERENCE_HASH_SOURCE, aggregate_scores, replay_gold_state_hash, score_sample
from .sdk import SessionFactory

InputMode = Literal["oracle", "asr"]


@dataclass(frozen=True)
class LabelReference:
    sample_id: str
    transcript: str | None
    tool_calls: tuple[Mapping[str, Any], ...]
    tool_results: tuple[Mapping[str, Any], ...]
    has_final_state: bool


@dataclass(frozen=True)
class EvaluationSummary:
    records_path: Path
    metrics_path: Path
    report_path: Path
    selected_ids: tuple[str, ...]
    aggregate: Mapping[str, Any]


def load_label_references(dataset: DatasetReader) -> Mapping[str, LabelReference]:
    if dataset.split != "train":
        raise DatasetError("labeled references are only available for the train split")
    references: dict[str, LabelReference] = {}
    with dataset.input_path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise DatasetError(f"{dataset.input_path}:{line_no}: expected object")
            sample_id = raw.get("id")
            if not isinstance(sample_id, str):
                raise DatasetError(f"{dataset.input_path}:{line_no}: missing id")
            if sample_id in references:
                raise DatasetError(f"{dataset.input_path}:{line_no}: duplicate id {sample_id!r}")
            label = raw.get("label")
            if not isinstance(label, Mapping):
                raise DatasetError(f"{dataset.input_path}:{line_no}: missing label object")
            raw_calls = label.get("tool_calls")
            raw_results = label.get("tool_results")
            if not isinstance(raw_calls, list) or not all(isinstance(item, Mapping) for item in raw_calls):
                raise DatasetError(f"{dataset.input_path}:{line_no}: label.tool_calls must be an array of objects")
            if not isinstance(raw_results, list) or not all(isinstance(item, Mapping) for item in raw_results):
                raise DatasetError(f"{dataset.input_path}:{line_no}: label.tool_results must be an array of objects")
            transcript = raw.get("audio_text")
            references[sample_id] = LabelReference(
                sample_id=sample_id,
                transcript=transcript if isinstance(transcript, str) and transcript.strip() else None,
                tool_calls=tuple(_normalize_call(item) for item in raw_calls),
                tool_results=tuple(dict(item) for item in raw_results),
                has_final_state=isinstance(label.get("final_state"), Mapping),
            )
    return references


def select_sample_ids(
    references: Mapping[str, LabelReference],
    *,
    sample_ids: Sequence[str] | None,
    limit: int | None,
    seed: int,
) -> tuple[str, ...]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if sample_ids:
        unique: list[str] = []
        seen: set[str] = set()
        for sample_id in sample_ids:
            if sample_id not in references:
                raise KeyError(f"unknown labeled sample id {sample_id!r}")
            if sample_id not in seen:
                seen.add(sample_id)
                unique.append(sample_id)
        return tuple(unique[:limit] if limit is not None else unique)

    ids = list(references)
    if limit is None:
        return tuple(ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return tuple(ids[:limit])


def run_semantic_evaluation(
    *,
    dataset: DatasetReader,
    references: Mapping[str, LabelReference],
    selected_ids: Sequence[str],
    runtime,
    decision_source_factory: Callable[[Sample], DecisionSource],
    output_dir: Path,
    input_mode: InputMode,
    sample_timeout_seconds: float | None,
    compute_reference_state_hash: bool = True,
    transcriber=None,
    reference_session_factory: SessionFactory | None = None,
    reference_hash_cache: dict[tuple[str, str, str], tuple[str | None, str | None]] | None = None,
    run_identity: Mapping[str, Any] | None = None,
) -> EvaluationSummary:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "evaluation-records.jsonl"
    metrics_path = output_dir / "evaluation-metrics.json"
    report_path = output_dir / "evaluation-report.md"
    run_config_path = output_dir / "run-config.json"

    catalog = dataset.load_catalog()
    tool_kind = {name: str(spec.get("kind", "unknown")) for name, spec in catalog.tools.items()}
    identity = dict(run_identity or {})
    identity.update(
        {
            "dataset_root": str(dataset.dataset_root),
            "split": dataset.split,
            "catalog_schema_version": catalog.schema_version,
            "selected_sample_count": len(selected_ids),
            "selected_ids_sha256": hashlib.sha256(
                "\n".join(selected_ids).encode("utf-8")
            ).hexdigest(),
        }
    )
    identity = _jsonable(identity)
    _write_json(run_config_path, identity)

    records: list[dict[str, Any]] = []
    confusion = Counter()
    argument_error_breakdown = Counter()

    progress_description = "evaluate"
    model = identity.get("model")
    output_mode = identity.get("agent_output_mode")
    if model and output_mode:
        progress_description = f"{model} {output_mode}/{input_mode}"
    selected_with_progress = tqdm(
        selected_ids,
        desc=progress_description,
        unit="sample",
        dynamic_ncols=True,
    )
    for index, sample_id in enumerate(selected_with_progress):
        sample = dataset.get_sample(sample_id)
        label = references[sample_id]
        deadline = SampleDeadline.start(sample_timeout_seconds) if sample_timeout_seconds is not None else None

        started = time.perf_counter()
        run_result: RunResult | None = None
        asr_latency = 0.0
        asr_turns = 0
        asr_hits = 0
        run_error: str | None = None

        try:
            decision_source = decision_source_factory(sample)
            set_deadline = getattr(decision_source, "set_deadline", None)
            if deadline is not None and callable(set_deadline):
                set_deadline(deadline)
            if input_mode == "oracle":
                if label.transcript is None:
                    raise ValueError("oracle transcript is unavailable for this sample")
                run_result = runtime.run(sample_id, label.transcript, decision_source, deadline=deadline)
            else:
                if transcriber is None:
                    raise ValueError("ASR mode requires a transcriber")
                audio_result = AudioInferencePipeline(dataset, runtime, transcriber).run(
                    sample_id,
                    decision_source,
                    deadline=deadline,
                )
                run_result = audio_result.run
                asr_latency = (
                    audio_result.diagnostics.final_audio_latency_seconds
                    + audio_result.diagnostics.history_audio_latency_seconds
                )
                asr_turns = audio_result.diagnostics.transcribed_turns
                asr_hits = audio_result.diagnostics.cache_hits
        except SampleDeadlineExceeded as exc:
            run_error = str(exc)
        except Exception as exc:  # noqa: BLE001 - evaluation must continue per sample.
            run_error = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - started
        if run_result is None:
            record = {
                "id": sample_id,
                "index": index,
                "input_mode": input_mode,
                "error": run_error or "unknown evaluation failure",
                "termination_reason": "sample_timeout" if isinstance(run_error, str)
                and "sample_timeout" in run_error else "evaluation_error",
                "sample_latency_seconds": elapsed,
                "expected_tool_calls": list(label.tool_calls),
                "predicted_tool_calls": None,
                "final_state_match": None,
                "metrics": None,
                "step_statuses": [],
                "sdk_rejected_call_indices": [],
            }
            records.append(record)
            continue

        expected_calls = [_normalize_call(item) for item in label.tool_calls]
        predicted_calls = [_normalize_call(item) for item in run_result.submission["tool_calls"]]
        comparison = _compare_calls(expected_calls, predicted_calls, tool_kind, catalog.tools)

        for pair, count in comparison["tool_confusions"].items():
            confusion[pair] += count
        for key, count in comparison["argument_error_breakdown"].items():
            argument_error_breakdown[key] += count

        usage = dict(run_result.model_usage or {})
        record = {
            "id": sample_id,
            "index": index,
            "input_mode": input_mode,
            "termination_reason": run_result.termination_reason,
            "sample_latency_seconds": elapsed,
            "asr_latency_seconds": asr_latency,
            "asr_turns": asr_turns,
            "asr_cache_hits": asr_hits,
            "model_latency_seconds": run_result.model_latency_seconds,
            "model_requests": int(usage.get("requests", 0)),
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
            "sdk_tool_calls": run_result.tool_calls_sent,
            "expected_tool_calls": expected_calls,
            "predicted_tool_calls": predicted_calls,
            "predicted_state_hash": run_result.submission["state_hash"],
            "step_statuses": [step.status for step in run_result.steps],
            "sdk_rejected_call_indices": _sdk_rejected_indices(run_result),
            "model_request_diagnostics": [dict(item) for item in run_result.model_request_diagnostics],
            "metrics": comparison,
        }
        records.append(record)

    # Gold/reference sessions exist only after this mode's entire inference pass.
    # The cache is evaluator-owned and never passed into a planner or Runtime.
    cache = reference_hash_cache if reference_hash_cache is not None else {}
    for record in records:
        sample_id = record["id"]
        expected_hash, reference_error = None, None
        if compute_reference_state_hash and record["predicted_tool_calls"] is not None:
            key = (str(dataset.dataset_root.resolve()), dataset.split, sample_id)
            if key not in cache:
                try:
                    cache[key] = (replay_gold_state_hash(
                        dataset.dataset_root, dataset.split, sample_id, references[sample_id].tool_calls,
                        session_factory=reference_session_factory,
                    ), None)
                except Exception as exc:  # noqa: BLE001 - mark reference unavailable, never assume equality.
                    cache[key] = (None, f"{type(exc).__name__}: {exc}")
            expected_hash, reference_error = cache[key]
        record.update(score_sample(
            references[sample_id].tool_calls, record["predicted_tool_calls"],
            record.get("predicted_state_hash"), expected_hash,
            rejected_indices=record["sdk_rejected_call_indices"],
        ))
        record.update({
            "expected_state_hash": expected_hash,
            "reference_state_hash_source": REFERENCE_HASH_SOURCE if compute_reference_state_hash else "disabled",
            "reference_state_hash_error": reference_error,
            "final_state_match": record["hash_match"],
        })

    _write_jsonl(records_path, records)
    aggregate = _aggregate(records, confusion, argument_error_breakdown)
    aggregate["run_identity"] = identity
    _write_json(metrics_path, aggregate)
    _write_markdown_report(report_path, aggregate, records)

    return EvaluationSummary(
        records_path=records_path,
        metrics_path=metrics_path,
        report_path=report_path,
        selected_ids=tuple(selected_ids),
        aggregate=aggregate,
    )


def _normalize_call(value: Mapping[str, Any]) -> dict[str, Any]:
    name = value.get("name")
    arguments = value.get("arguments", {})
    if not isinstance(name, str):
        raise ValueError("tool call name must be a string")
    if not isinstance(arguments, Mapping):
        raise ValueError("tool call arguments must be an object")
    return {"name": name, "arguments": dict(arguments)}


def _compare_calls(
    expected_calls: Sequence[Mapping[str, Any]],
    predicted_calls: Sequence[Mapping[str, Any]],
    tool_kind: Mapping[str, str],
    catalog_tools: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_names = [item["name"] for item in expected_calls]
    predicted_names = [item["name"] for item in predicted_calls]
    aligned = min(len(expected_calls), len(predicted_calls))
    positional_name_matches = sum(1 for i in range(aligned) if expected_names[i] == predicted_names[i])
    name_denominator = max(len(expected_calls), 1)

    argument_exact = 0
    strict_field_total = 0
    strict_field_correct = 0
    relaxed_field_correct = 0
    missing_required = 0
    unexpected_arguments = 0
    argument_error_breakdown: Counter[str] = Counter()

    tool_confusions: Counter[str] = Counter()
    for index in range(max(len(expected_calls), len(predicted_calls))):
        expected_name = expected_names[index] if index < len(expected_names) else "<none>"
        predicted_name = predicted_names[index] if index < len(predicted_names) else "<none>"
        if expected_name != predicted_name:
            tool_confusions[f"{expected_name} -> {predicted_name}"] += 1

    for index in range(aligned):
        expected = expected_calls[index]
        predicted = predicted_calls[index]
        expected_name = str(expected["name"])
        predicted_name = str(predicted["name"])
        expected_args = dict(expected["arguments"])
        predicted_args = dict(predicted["arguments"])

        if expected_name == predicted_name and expected_args == predicted_args:
            argument_exact += 1

        strict_field_total += len(expected_args)
        required_keys = _required_fields(catalog_tools.get(expected_name))
        missing_required += len([key for key in required_keys if key not in predicted_args])
        unexpected = set(predicted_args) - set(expected_args)
        unexpected_arguments += len(unexpected)

        for key, expected_value in expected_args.items():
            predicted_has_key = key in predicted_args
            predicted_value = predicted_args.get(key)
            if predicted_has_key and predicted_value == expected_value:
                strict_field_correct += 1
                relaxed_field_correct += 1
                continue
            if predicted_has_key and _equivalent_value(expected_value, predicted_value):
                relaxed_field_correct += 1
                argument_error_breakdown["strict_mismatch_equivalent"] += 1
                continue
            category = _argument_error_category(key, expected_value, predicted_value, predicted_has_key)
            argument_error_breakdown[category] += 1

    missing_calls = max(0, len(expected_calls) - len(predicted_calls))
    extra_calls = max(0, len(predicted_calls) - len(expected_calls))
    edit_distance = _levenshtein(expected_names, predicted_names)

    repeated_calls = sum(
        count - 1 for count in Counter(_call_key(item) for item in predicted_calls).values() if count > 1
    )
    excessive_reads = _excessive_reads(expected_calls, predicted_calls, tool_kind)

    return {
        "tool_name_accuracy": positional_name_matches / name_denominator,
        "tool_sequence_exact_match": expected_calls == predicted_calls,
        "tool_sequence_edit_distance": edit_distance,
        "missing_calls": missing_calls,
        "extra_calls": extra_calls,
        "incorrect_app_or_tool_count": sum(1 for i in range(aligned) if expected_names[i] != predicted_names[i]),
        "argument_exact_match_accuracy": argument_exact / max(aligned, 1),
        "argument_field_accuracy": strict_field_correct / max(strict_field_total, 1),
        "argument_field_accuracy_relaxed": relaxed_field_correct / max(strict_field_total, 1),
        "missing_required_arguments": missing_required,
        "unexpected_arguments": unexpected_arguments,
        "repeated_or_unnecessary_calls": repeated_calls,
        "excessive_read_calls": excessive_reads,
        "tool_confusions": dict(sorted(tool_confusions.items())),
        "argument_error_breakdown": dict(sorted(argument_error_breakdown.items())),
    }


def _required_fields(spec: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(spec, Mapping):
        return set()
    params = spec.get("parameters")
    if not isinstance(params, Mapping):
        return set()
    required = params.get("required")
    if not isinstance(required, list):
        return set()
    return {item for item in required if isinstance(item, str)}


def _equivalent_value(expected: Any, predicted: Any) -> bool:
    if expected == predicted:
        return True
    if isinstance(expected, str) and isinstance(predicted, str):
        return expected.strip().casefold() == predicted.strip().casefold()
    if isinstance(expected, (int, float)) and isinstance(predicted, (int, float)):
        return math.isclose(float(expected), float(predicted), rel_tol=0.0, abs_tol=1e-9)
    return False


def _argument_error_category(key: str, expected: Any, predicted: Any, predicted_has_key: bool) -> str:
    lowered = key.casefold()
    if not predicted_has_key:
        return "missing_argument"
    if any(token in lowered for token in ("date", "time", "hour", "minute", "timezone")):
        return "datetime_mismatch"
    if any(token in lowered for token in ("amount", "price", "balance", "money", "fare")):
        return "money_or_numeric_mismatch"
    if any(token in lowered for token in ("id", "name", "contact", "person")):
        return "name_or_identifier_mismatch"
    if any(token in lowered for token in ("address", "location", "origin", "destination", "pickup", "dropoff")):
        return "location_mismatch"
    if isinstance(expected, (int, float)) or isinstance(predicted, (int, float)):
        return "numeric_mismatch"
    return "value_mismatch"


def _call_key(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _excessive_reads(
    expected_calls: Sequence[Mapping[str, Any]],
    predicted_calls: Sequence[Mapping[str, Any]],
    tool_kind: Mapping[str, str],
) -> int:
    expected_counts = Counter(
        call["name"] for call in expected_calls if tool_kind.get(str(call["name"])) == "read"
    )
    predicted_counts = Counter(
        call["name"] for call in predicted_calls if tool_kind.get(str(call["name"])) == "read"
    )
    excessive = 0
    for name, count in predicted_counts.items():
        excessive += max(0, count - expected_counts.get(name, 0))
    return excessive


def _levenshtein(left: Sequence[str], right: Sequence[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, left_value in enumerate(left, 1):
        current = [i]
        for j, right_value in enumerate(right, 1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (0 if left_value == right_value else 1)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _sdk_rejected_indices(result: RunResult) -> list[int]:
    # Only dispatched steps correspond to SDK trace entries. Local validation,
    # permission and duplicate rejections must not be added to the submission.
    dispatched_statuses = {
        "sdk_rejected", "tool_error", "no_op", "write_succeeded", "read_succeeded",
        "meta_succeeded", "executed", "execution_uncertain",
    }
    dispatched = [step for step in result.steps if step.status in dispatched_statuses]
    return [index for index, step in enumerate(dispatched) if step.status == "sdk_rejected"]


def _aggregate(
    records: Sequence[Mapping[str, Any]],
    confusion: Counter[str],
    argument_breakdown: Counter[str],
) -> dict[str, Any]:
    successful = [record for record in records if record.get("metrics") is not None]
    failed = [record for record in records if record.get("metrics") is None]
    if successful:
        metric_rows = [record["metrics"] for record in successful]
        latencies = [float(record["sample_latency_seconds"]) for record in successful]
        model_latencies = [
            float(record["model_latency_seconds"])
            for record in successful
            if record.get("model_latency_seconds") is not None
        ]
        asr_latencies = [float(record.get("asr_latency_seconds", 0.0)) for record in successful]
        asr_turns = sum(int(record.get("asr_turns", 0)) for record in successful)
        asr_cache_hits = sum(int(record.get("asr_cache_hits", 0)) for record in successful)
        model_requests = [int(record.get("model_requests", 0)) for record in successful]
        sdk_calls = [int(record.get("sdk_tool_calls", 0)) for record in successful]
        input_tokens = sum(int(record.get("input_tokens", 0)) for record in successful)
        output_tokens = sum(int(record.get("output_tokens", 0)) for record in successful)
        total_tokens = sum(int(record.get("total_tokens", 0)) for record in successful)
        final_state_matches = [record.get("final_state_match") for record in successful]

        aggregate = {
            "evaluated_samples": len(records),
            "successful_samples": len(successful),
            "failed_samples": len(failed),
            "tool_name_accuracy": _mean(metric["tool_name_accuracy"] for metric in metric_rows),
            "tool_sequence_exact_match_rate": _mean(
                1.0 if metric["tool_sequence_exact_match"] else 0.0 for metric in metric_rows
            ),
            "tool_sequence_edit_distance_mean": _mean(
                metric["tool_sequence_edit_distance"] for metric in metric_rows
            ),
            "argument_exact_match_accuracy": _mean(
                metric["argument_exact_match_accuracy"] for metric in metric_rows
            ),
            "argument_field_accuracy": _mean(metric["argument_field_accuracy"] for metric in metric_rows),
            "argument_field_accuracy_relaxed": _mean(
                metric["argument_field_accuracy_relaxed"] for metric in metric_rows
            ),
            "missing_required_arguments": int(sum(metric["missing_required_arguments"] for metric in metric_rows)),
            "unexpected_arguments": int(sum(metric["unexpected_arguments"] for metric in metric_rows)),
            "missing_calls": int(sum(metric["missing_calls"] for metric in metric_rows)),
            "extra_calls": int(sum(metric["extra_calls"] for metric in metric_rows)),
            "incorrect_app_or_tool_count": int(
                sum(metric["incorrect_app_or_tool_count"] for metric in metric_rows)
            ),
            "repeated_or_unnecessary_calls": int(
                sum(metric["repeated_or_unnecessary_calls"] for metric in metric_rows)
            ),
            "excessive_read_calls": int(sum(metric["excessive_read_calls"] for metric in metric_rows)),
            "mean_latency_seconds": _mean(latencies),
            "p95_latency_seconds": _percentile(latencies, 95),
            "mean_model_latency_seconds": _mean(model_latencies),
            "p95_model_latency_seconds": _percentile(model_latencies, 95),
            "mean_asr_latency_seconds": _mean(asr_latencies),
            "total_asr_latency_seconds": float(sum(asr_latencies)),
            "asr_turns": asr_turns,
            "asr_cache_hits": asr_cache_hits,
            "asr_cache_hit_rate": asr_cache_hits / asr_turns if asr_turns else None,
            "model_requests_per_sample": _mean(model_requests),
            "sdk_calls_per_sample": _mean(sdk_calls),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "tokens_per_sample": total_tokens / len(successful),
            "final_state_accuracy": _state_accuracy(final_state_matches),
            "timeout_count": int(
                sum(1 for record in successful if record.get("termination_reason") == "sample_timeout")
                + sum(1 for record in failed if "sample_timeout" in str(record.get("error", "")))
            ),
            "failure_rate": len(failed) / max(len(records), 1),
            "tool_confusion": dict(confusion.most_common(50)),
            "argument_error_breakdown": dict(argument_breakdown),
            "worst_samples": _worst_samples(records),
        }
    else:
        aggregate = {
            "evaluated_samples": len(records),
            "successful_samples": 0,
            "failed_samples": len(failed),
            "failure_rate": 1.0 if records else 0.0,
            "tool_confusion": {},
            "argument_error_breakdown": {},
            "worst_samples": _worst_samples(records),
        }
    aggregate.update(aggregate_scores(records))
    aggregate.update(_execution_diagnostics(records))
    return aggregate


def _execution_diagnostics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep controller quality separate from missing result and scoring gates.

    Rates are fractions of ALL evaluated samples, not fractions of steps. A
    sample can have several error categories. A missing result is not also
    counted as an abnormal termination of an available result.
    """
    normal = {"model_completed", "awaiting_user", "unsupported_request", "decision_source_done"}
    malformed = {"model_bad_output", "model_bad_output_limit"}
    steps = Counter(status for row in records for status in row.get("step_statuses", []))
    total = len(records)
    missing = sum(row.get("metrics") is None for row in records)
    abnormal = sum(
        row.get("metrics") is not None and row.get("termination_reason") not in normal for row in records
    )
    bad_output = sum(bool(malformed.intersection(row.get("step_statuses", []))) for row in records)
    schema_samples = sum("schema_rejected" in row.get("step_statuses", []) for row in records)
    finish_samples = sum("invalid_finish" in row.get("step_statuses", []) for row in records)
    timeout_samples = sum(
        row.get("termination_reason") == "sample_timeout" or "sample_timeout" in str(row.get("error", ""))
        for row in records
    )
    return {
        "reference_state_hash_source": REFERENCE_HASH_SOURCE,
        "reference_hash_failure_samples": sum(bool(row.get("reference_state_hash_error")) for row in records),
        "infrastructure_failure_samples": missing,
        "infrastructure_failure_rate": missing / total if total else 0.0,
        "abnormal_termination_samples": abnormal,
        "abnormal_termination_rate": abnormal / total if total else 0.0,
        "malformed_output_proposals": sum(steps[status] for status in malformed),
        "malformed_output_samples": bad_output,
        "invalid_output_rate": bad_output / total if total else 0.0,
        "schema_rejected_proposals": steps["schema_rejected"],
        "schema_rejection_samples": schema_samples,
        "schema_rejection_sample_rate": schema_samples / total if total else 0.0,
        "invalid_finish_proposals": steps["invalid_finish"],
        "invalid_finish_samples": finish_samples,
        "invalid_finish_sample_rate": finish_samples / total if total else 0.0,
        "timeout_count": timeout_samples,
        "timeout_rate": timeout_samples / total if total else 0.0,
        "termination_reasons": dict(Counter(row.get("termination_reason", "evaluation_error") for row in records)),
        "step_status_counts": dict(steps),
    }


def _state_accuracy(values: Sequence[Any]) -> float | None:
    available = [item for item in values if isinstance(item, bool)]
    if not available:
        return None
    return sum(1 for item in available if item) / len(available)


def _worst_samples(records: Sequence[Mapping[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    ranked: list[tuple[float, Mapping[str, Any]]] = []
    for record in records:
        metrics = record.get("metrics")
        if not isinstance(metrics, Mapping):
            ranked.append((0.0, record))
            continue
        score = float(record["challenge_score_estimate"])
        ranked.append((score, record))
    ranked.sort(key=lambda item: (item[0], str(item[1].get("id"))))
    rows: list[dict[str, Any]] = []
    for score, record in ranked[:limit]:
        rows.append(
            {
                "id": record.get("id"),
                "score": score,
                "score_reason": record.get("score_reason"),
                "termination_reason": record.get("termination_reason"),
                "error": record.get("error"),
                "expected": record.get("expected_tool_calls"),
                "predicted": record.get("predicted_tool_calls"),
            }
        )
    return rows


def _mean(values: Sequence[float] | Sequence[int]) -> float:
    items = list(values)
    return float(sum(items) / len(items)) if items else 0.0


def _percentile(values: Sequence[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    content = "".join(json.dumps(_jsonable(row), ensure_ascii=False) + "\n" for row in rows)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown_report(path: Path, aggregate: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> None:
    identity = aggregate.get("run_identity", {})
    lines = [
        "# Semantic Evaluation Report",
        "",
        "## Challenge score estimate (primary; not official-equivalent)",
        "",
        f"- Challenge score estimate (0–100): {_fmt_float(aggregate.get('challenge_score_estimate'))}",
        "- Matching: strict canonical JSON, exact names, unordered one-to-one; no fuzzy/partial matches.",
        "- Official call matcher unavailable; challenge_score_exact=false.",
        "- Gold hashes: isolated offline SDK gold-call replay; full-state hashes, not partial label.final_state.",
        f"- Reference hash errors: {aggregate['reference_hash_failure_samples']}",
        f"- Perfect / partial / zero: {aggregate['perfect_score_samples']} / "
        f"{aggregate['partial_score_samples']} / {aggregate['zero_score_samples']}",
        f"- Hash-match rate (all samples): {_fmt_float(aggregate['hash_match_rate'])}",
        f"- Hash match / mismatch / unavailable: {aggregate['hash_match_samples']} / "
        f"{aggregate['hash_mismatch_samples']} / {aggregate['hash_unavailable_samples']}",
        f"- Calls gold / submitted / matched / unmatched: {aggregate['gold_calls']} / "
        f"{aggregate['submitted_calls']} / {aggregate['matched_calls']} / {aggregate['unmatched_calls']}",
        f"- SDK-rejected submitted calls: {aggregate['schema_rejected_submitted_calls']}",
        f"- Zero due to hash mismatch: {aggregate['zero_due_to_hash_mismatch']}",
        f"- Zero with no matched calls: {aggregate['zero_due_to_no_matched_calls']}",
        "- These zero-score breakdowns can overlap (both gates may fail).",
        f"- Empty gold / perfect empty submission / nonempty submission: {aggregate['empty_gold_samples']} / "
        f"{aggregate['empty_gold_perfect_samples']} / {aggregate['empty_gold_nonempty_submission_samples']}",
        f"- Scoring reasons: {json.dumps(aggregate['score_reasons'], sort_keys=True)}",
        "- Score-affecting: hash gate, unmatched submitted calls, missed gold calls, unusable predictions.",
        "- Diagnostic only: positional accuracy, relaxed argument accuracy, latency and local rejected proposals.",
        "- All evaluated samples, including failures, enter the primary denominator; "
        "diagnostics below retain their existing successful-run denominator.",
        "",
        "## Summary",
        "",
        f"- Model: {identity.get('model', 'unknown') if isinstance(identity, Mapping) else 'unknown'}",
        f"- Agent output mode: "
        f"{identity.get('agent_output_mode', 'unknown') if isinstance(identity, Mapping) else 'unknown'}",
        f"- Input mode: {identity.get('input_mode', 'unknown') if isinstance(identity, Mapping) else 'unknown'}",
        f"- Evaluated samples: {aggregate.get('evaluated_samples', 0)}",
        f"- Samples with usable results (legacy successful_samples): {aggregate.get('successful_samples', 0)}",
        f"- Infrastructure/missing-result failures: {aggregate['infrastructure_failure_samples']}",
        f"- Abnormal terminations with results: {aggregate['abnormal_termination_samples']}",
        f"- Malformed output proposals / affected samples: {aggregate['malformed_output_proposals']} / "
        f"{aggregate['malformed_output_samples']}",
        f"- Locally schema-rejected proposals / affected samples: {aggregate['schema_rejected_proposals']} / "
        f"{aggregate['schema_rejection_samples']}",
        f"- Invalid finish proposals / affected samples: {aggregate['invalid_finish_proposals']} / "
        f"{aggregate['invalid_finish_samples']}",
        f"- Timeout samples / rate: {aggregate['timeout_count']} / {_fmt_float(aggregate['timeout_rate'])}",
        "- Error sample rates use all evaluated samples; error categories may overlap.",
        f"- Tool-name accuracy: {_fmt_float(aggregate.get('tool_name_accuracy'))}",
        f"- Tool-sequence exact match: {_fmt_float(aggregate.get('tool_sequence_exact_match_rate'))}",
        f"- Argument field accuracy: {_fmt_float(aggregate.get('argument_field_accuracy'))}",
        f"- Final-state accuracy: {_fmt_float_or_na(aggregate.get('final_state_accuracy'))}",
        f"- Mean sample wall latency (s; ASR-cache-sensitive): "
        f"{_fmt_float(aggregate.get('mean_latency_seconds'))}",
        f"- P95 sample wall latency (s; ASR-cache-sensitive): "
        f"{_fmt_float(aggregate.get('p95_latency_seconds'))}",
        f"- Mean model latency (s): {_fmt_float(aggregate.get('mean_model_latency_seconds'))}",
        f"- P95 model latency (s): {_fmt_float(aggregate.get('p95_model_latency_seconds'))}",
        f"- ASR cache hits / turns: {aggregate.get('asr_cache_hits', 0)} / "
        f"{aggregate.get('asr_turns', 0)}",
        f"- Input / output / total model tokens: {aggregate.get('input_tokens', 0)} / "
        f"{aggregate.get('output_tokens', 0)} / {aggregate.get('total_tokens', 0)}",
        "",
        "## Worst Samples",
        "",
    ]

    worst = aggregate.get("worst_samples")
    if isinstance(worst, list) and worst:
        for row in worst[:10]:
            lines.append(
                f"- {row.get('id')}: score={_fmt_float(row.get('score'))} scoring={row.get('score_reason')} "
                f"reason={row.get('termination_reason')} error={row.get('error')}; "
                f"expected={json.dumps(row.get('expected'), ensure_ascii=False)}; "
                f"predicted={json.dumps(row.get('predicted'), ensure_ascii=False)}"
            )
    else:
        lines.append("- N/A")

    lines.extend(
        [
            "",
            "## Tool Confusion",
            "",
        ]
    )
    confusion = aggregate.get("tool_confusion")
    if isinstance(confusion, Mapping) and confusion:
        for pair, count in list(confusion.items())[:20]:
            lines.append(f"- {pair}: {count}")
    else:
        lines.append("- N/A")

    lines.extend(
        [
            "",
            "## Argument Error Breakdown",
            "",
        ]
    )
    argument_breakdown = aggregate.get("argument_error_breakdown")
    if isinstance(argument_breakdown, Mapping) and argument_breakdown:
        for key, count in argument_breakdown.items():
            lines.append(f"- {key}: {count}")
    else:
        lines.append("- N/A")

    lines.extend(
        [
            "",
            "## Sample Failures",
            "",
        ]
    )
    failures = [record for record in records if record.get("metrics") is None or
                record.get("termination_reason") not in {
                    "model_completed", "awaiting_user", "unsupported_request", "decision_source_done"
                } or record.get("reference_state_hash_error")]
    if failures:
        for row in failures[:20]:
            lines.append(
                f"- {row.get('id')}: termination={row.get('termination_reason')}; "
                f"error={row.get('error')}; reference_error={row.get('reference_state_hash_error')}"
            )
    else:
        lines.append("- None")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "0.0000"


def _fmt_float_or_na(value: Any) -> str:
    if value is None:
        return "N/A"
    return _fmt_float(value)


def compare_aggregates(native_metrics: Mapping[str, Any], constrained_metrics: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    metrics = [
        ("Challenge score estimate (0–100)", "challenge_score_estimate"),
        ("Hash match rate", "hash_match_rate"),
        ("Perfect sample rate", "perfect_sample_rate"),
        ("Partial sample rate", "partial_sample_rate"),
        ("Tool-name accuracy", "tool_name_accuracy"),
        ("Tool-sequence exact match", "tool_sequence_exact_match_rate"),
        ("Argument accuracy", "argument_field_accuracy"),
        ("Final-state accuracy", "final_state_accuracy"),
        ("Invalid output sample rate", "invalid_output_rate"),
        ("Schema-rejection sample rate", "schema_rejection_sample_rate"),
        ("Invalid-finish sample rate", "invalid_finish_sample_rate"),
        ("Abnormal termination rate", "abnormal_termination_rate"),
        ("Model requests/sample", "model_requests_per_sample"),
        ("SDK calls/sample", "sdk_calls_per_sample"),
        ("Model tokens/sample", "tokens_per_sample"),
        ("Mean model latency", "mean_model_latency_seconds"),
        ("P95 model latency", "p95_model_latency_seconds"),
        ("Mean sample wall latency (ASR-cache-sensitive)", "mean_latency_seconds"),
        ("P95 sample wall latency (ASR-cache-sensitive)", "p95_latency_seconds"),
        ("Mean ASR latency", "mean_asr_latency_seconds"),
        ("ASR cache-hit rate", "asr_cache_hit_rate"),
        ("Timeout rate", "timeout_rate"),
        ("Infrastructure failure rate", "infrastructure_failure_rate"),
    ]
    for label, key in metrics:
        native = native_metrics.get(key)
        candidate = constrained_metrics.get(key)
        delta = _numeric_delta(native, candidate)
        rows.append(
            {
                "metric": label,
                "native": native,
                "constrained_json": candidate,
                "delta": delta,
            }
        )
    return {
        "sample_count_native": native_metrics.get("evaluated_samples"),
        "sample_count_constrained_json": constrained_metrics.get("evaluated_samples"),
        "rows": rows,
        "challenge_score_exact": False,
    }


def compare_evaluations(native: EvaluationSummary, constrained: EvaluationSummary) -> dict[str, Any]:
    """Compare paired records, rejecting missing, duplicate or unpaired IDs."""
    if native.selected_ids != constrained.selected_ids:
        raise ValueError("A/B evaluation must use identical ordered sample IDs")
    def read_records(summary: EvaluationSummary) -> dict[str, Any]:
        rows = [
            json.loads(line) for line in summary.records_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        ids = tuple(row["id"] for row in rows)
        if ids != summary.selected_ids or len(set(ids)) != len(ids):
            raise ValueError("evaluation records do not match unique selected sample IDs")
        return {row["id"]: row for row in rows}
    left, right = read_records(native), read_records(constrained)
    comparison = compare_aggregates(native.aggregate, constrained.aggregate)
    transitions: dict[str, list[str]] = {
        "improved_zero_to_nonzero": [], "regressed_nonzero_to_zero": [],
        "became_perfect": [], "lost_correct_state_hash": [],
    }
    for sample_id in native.selected_ids:
        before, after = left[sample_id], right[sample_id]
        a, b = before["challenge_score_estimate"], after["challenge_score_estimate"]
        if a == 0 and b > 0:
            transitions["improved_zero_to_nonzero"].append(sample_id)
        if a > 0 and b == 0:
            transitions["regressed_nonzero_to_zero"].append(sample_id)
        if a < 1 and b == 1:
            transitions["became_perfect"].append(sample_id)
        if before["hash_match"] is True and after["hash_match"] is not True:
            transitions["lost_correct_state_hash"].append(sample_id)
    comparison["sample_transitions"] = transitions
    return comparison


def write_comparison_markdown(path: Path, comparison: Mapping[str, Any]) -> None:
    lines = [
        "# Native vs Constrained JSON Comparison",
        "",
        f"- Native sample count: {comparison.get('sample_count_native')}",
        f"- Constrained sample count: {comparison.get('sample_count_constrained_json')}",
        "",
        "| Metric | Native | Constrained JSON | Delta |",
        "|---|---:|---:|---:|",
    ]
    rows = comparison.get("rows")
    if isinstance(rows, list):
        for row in rows:
            lines.append(
                "| "
                + f"{row.get('metric')} | {_fmt_value(row.get('native'))} | "
                f"{_fmt_value(row.get('constrained_json'))} | {_fmt_value(row.get('delta'))} |"
            )
    lines.extend([
        "",
        "Estimated matching only; not official-equivalent.",
        "Sample wall and ASR latency comparisons are cache-order-sensitive; compare model latency for A/B timing.",
        "",
        "## Paired sample transitions",
        "",
    ])
    for name, ids in comparison.get("sample_transitions", {}).items():
        lines.append(f"- {name} ({len(ids)}): {', '.join(ids) or 'None'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _numeric_delta(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return float(right) - float(left)


def _fmt_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
