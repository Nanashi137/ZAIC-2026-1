from __future__ import annotations

import json
from pathlib import Path

import pytest

from lifemate.agent_benchmark_report import summarize_agent_benchmark


def _metrics(root: Path, variant: str, *, sample_hash: str = "paired") -> None:
    path = root / variant / "native-oracle" / "evaluation-metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "challenge_score_estimate": 25.0,
                "hash_match_rate": 0.4,
                "tool_sequence_exact_match_rate": 0.2,
                "sdk_calls_per_sample": 2.0,
                "extra_calls": 10,
                "invalid_output_rate": 0.01,
                "timeout_rate": 0.0,
                "mean_latency_seconds": 5.0,
                "termination_reasons": {"model_completed": 100},
                "run_identity": {
                    "agent_architecture": "direct",
                    "prompt_version": "grounded-v2",
                    "invalid_response_history": "compact",
                    "input_mode": "oracle",
                    "selected_ids_sha256": sample_hash,
                },
            }
        ),
        encoding="utf-8",
    )
    scores = [0.2, 0.8] if variant == "baseline-direct" else [0.4, 0.9]
    path.with_name("evaluation-records.jsonl").write_text(
        "".join(
            json.dumps({"id": f"train_{index}", "challenge_score_estimate": score}) + "\n"
            for index, score in enumerate(scores)
        ),
        encoding="utf-8",
    )


def test_agent_benchmark_summary_requires_and_reports_paired_variants(tmp_path: Path) -> None:
    _metrics(tmp_path, "baseline-direct")
    _metrics(tmp_path, "candidate")

    json_path, markdown_path = summarize_agent_benchmark(tmp_path)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["variant_count"] == 2
    assert summary["selected_ids_sha256"] == "paired"
    assert [item["variant"] for item in summary["variants"]] == ["baseline-direct", "candidate"]
    assert summary["paired_vs_baseline"][0]["mean_delta_points"] == pytest.approx(15.0)
    assert summary["paired_vs_baseline"][0]["identical_trace_samples"] == 2
    report = markdown_path.read_text(encoding="utf-8")
    assert "| baseline-direct | direct | grounded-v2 | compact |" in report
    assert "## Paired score changes versus baseline" in report


def test_agent_benchmark_summary_rejects_unpaired_samples(tmp_path: Path) -> None:
    _metrics(tmp_path, "baseline-direct", sample_hash="first")
    _metrics(tmp_path, "candidate", sample_hash="second")

    with pytest.raises(ValueError, match="paired sample set"):
        summarize_agent_benchmark(tmp_path)
