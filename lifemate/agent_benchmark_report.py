"""Summarize fixed-subset agent architecture benchmark outputs."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

_METRICS = (
    "challenge_score_estimate",
    "perfect_sample_rate",
    "partial_sample_rate",
    "hash_match_rate",
    "tool_sequence_exact_match_rate",
    "tool_name_accuracy",
    "sdk_calls_per_sample",
    "extra_calls",
    "missing_calls",
    "excessive_read_calls",
    "unexpected_arguments",
    "missing_required_arguments",
    "invalid_output_rate",
    "timeout_rate",
    "mean_latency_seconds",
    "p95_latency_seconds",
    "model_requests_per_sample",
    "tokens_per_sample",
)


def summarize_agent_benchmark(root: Path) -> tuple[Path, Path]:
    root = root.expanduser().resolve()
    variants: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/native-*/evaluation-metrics.json")):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        identity = dict(metrics.get("run_identity", {}))
        variants.append(
            {
                "variant": metrics_path.parent.parent.name,
                "metrics_path": str(metrics_path),
                "agent_architecture": identity.get("agent_architecture"),
                "prompt_version": identity.get("prompt_version"),
                "invalid_response_history": identity.get("invalid_response_history"),
                "input_mode": identity.get("input_mode"),
                "selected_ids_sha256": identity.get("selected_ids_sha256"),
                **{name: metrics.get(name) for name in _METRICS},
                "termination_reasons": metrics.get("termination_reasons", {}),
            }
        )
    if not variants:
        raise ValueError(f"no agent benchmark metrics found beneath {root}")
    selected_hashes = {item["selected_ids_sha256"] for item in variants}
    input_modes = {item["input_mode"] for item in variants}
    if len(selected_hashes) != 1 or len(input_modes) != 1:
        raise ValueError("agent benchmark variants do not use one paired sample set and input mode")
    paired_comparisons = _paired_comparisons(root, variants)

    json_path = root / "agent-benchmark-summary.json"
    markdown_path = root / "agent-benchmark-summary.md"
    summary = {
        "benchmark_schema_version": 1,
        "variant_count": len(variants),
        "input_mode": next(iter(input_modes)),
        "selected_ids_sha256": next(iter(selected_hashes)),
        "variants": variants,
        "paired_vs_baseline": paired_comparisons,
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Agent benchmark summary",
        "",
        "| Variant | Architecture | Prompt | Invalid history | Score estimate | Hash match | Exact sequence | "
        "Calls/sample | Extra calls | Invalid output | Timeout | Mean latency |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in variants:
        lines.append(
            "| {variant} | {agent_architecture} | {prompt_version} | {invalid_response_history} | "
            "{challenge_score_estimate} | {hash_match_rate} | {tool_sequence_exact_match_rate} | "
            "{sdk_calls_per_sample} | {extra_calls} | {invalid_output_rate} | {timeout_rate} | "
            "{mean_latency_seconds} |".format(**item)
        )
    lines.extend(
        [
            "",
            "## Paired score changes versus baseline",
            "",
            "| Variant | Delta points | Bootstrap 95% interval | Improved | Worsened | Equal | "
            "Same first | Same trace | Same termination | Same hash |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in paired_comparisons:
        lines.append(
            "| {variant} | {mean_delta_points:.3f} | [{bootstrap_95_low_points:.3f}, "
            "{bootstrap_95_high_points:.3f}] | {improved_samples} | {worsened_samples} | "
            "{equal_samples} | {identical_first_action_samples}/{sample_count} | "
            "{identical_trace_samples}/{sample_count} | {identical_termination_samples}/{sample_count} | "
            "{identical_state_hash_samples}/{sample_count} |".format(**item)
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _paired_comparisons(root: Path, variants: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((item for item in variants if item["variant"] == "baseline-direct"), None)
    if baseline is None:
        raise ValueError("agent benchmark requires a baseline-direct variant")
    records_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for item in variants:
        metrics_path = Path(item["metrics_path"])
        records_path = metrics_path.with_name("evaluation-records.jsonl")
        if not records_path.is_file():
            raise ValueError(f"missing paired evaluation records: {records_path}")
        records: dict[str, dict[str, Any]] = {}
        for line in records_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records[str(row["id"])] = row
        records_by_variant[str(item["variant"])] = records
    baseline_records = records_by_variant["baseline-direct"]
    comparisons: list[dict[str, Any]] = []
    for item in variants:
        variant = str(item["variant"])
        if variant == "baseline-direct":
            continue
        candidate = records_by_variant[variant]
        if candidate.keys() != baseline_records.keys():
            raise ValueError(f"variant {variant!r} does not contain the baseline sample IDs")
        deltas = [
            float(candidate[sample_id]["challenge_score_estimate"])
            - float(baseline_records[sample_id]["challenge_score_estimate"])
            for sample_id in baseline_records
        ]
        bootstrap = _bootstrap_means(deltas, seed=42, draws=20_000)
        sample_count = len(baseline_records)
        comparisons.append(
            {
                "variant": variant,
                "mean_delta_points": 100 * sum(deltas) / len(deltas),
                "bootstrap_95_low_points": 100 * bootstrap[int(0.025 * len(bootstrap))],
                "bootstrap_95_high_points": 100 * bootstrap[int(0.975 * len(bootstrap)) - 1],
                "improved_samples": sum(delta > 1e-12 for delta in deltas),
                "worsened_samples": sum(delta < -1e-12 for delta in deltas),
                "equal_samples": sum(abs(delta) <= 1e-12 for delta in deltas),
                "sample_count": sample_count,
                "identical_first_action_samples": sum(
                    _first_action(candidate[sample_id]) == _first_action(baseline_records[sample_id])
                    for sample_id in baseline_records
                ),
                "identical_trace_samples": sum(
                    candidate[sample_id].get("predicted_tool_calls", [])
                    == baseline_records[sample_id].get("predicted_tool_calls", [])
                    for sample_id in baseline_records
                ),
                "identical_termination_samples": sum(
                    candidate[sample_id].get("termination_reason")
                    == baseline_records[sample_id].get("termination_reason")
                    for sample_id in baseline_records
                ),
                "identical_state_hash_samples": sum(
                    candidate[sample_id].get("predicted_state_hash")
                    == baseline_records[sample_id].get("predicted_state_hash")
                    for sample_id in baseline_records
                ),
            }
        )
    return comparisons


def _first_action(row: dict[str, Any]) -> Any:
    calls = row.get("predicted_tool_calls", [])
    return calls[0] if calls else None


def _bootstrap_means(values: list[float], *, seed: int, draws: int) -> list[float]:
    if not values:
        raise ValueError("paired comparison requires at least one score")
    rng = random.Random(seed)
    count = len(values)
    return sorted(
        sum(values[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(draws)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    json_path, markdown_path = summarize_agent_benchmark(args.root)
    print(json.dumps({"summary": str(json_path), "report": str(markdown_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
