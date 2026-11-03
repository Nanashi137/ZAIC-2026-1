"""Summarize fixed-subset agent architecture benchmark outputs."""

from __future__ import annotations

import argparse
import json
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

    json_path = root / "agent-benchmark-summary.json"
    markdown_path = root / "agent-benchmark-summary.md"
    summary = {
        "benchmark_schema_version": 1,
        "variant_count": len(variants),
        "input_mode": next(iter(input_modes)),
        "selected_ids_sha256": next(iter(selected_hashes)),
        "variants": variants,
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
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    json_path, markdown_path = summarize_agent_benchmark(args.root)
    print(json.dumps({"summary": str(json_path), "report": str(markdown_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
