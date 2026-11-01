"""Offline multi-sample Phase 04 smoke with scripted models and the real SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifemate.agent import LlmDecisionSource
from lifemate.batch import BatchConfig, BatchRunner
from lifemate.data import DatasetReader
from lifemate.model import InferenceSettings, ModelResponse, ScriptedModelClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/train+public_test"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--full", action="store_true", help="Process the complete public-test split offline.")
    parser.add_argument("--output", type=Path, default=Path("outputs/phase04-smoke/submission.jsonl"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    dataset = DatasetReader(args.dataset_root)

    def decisions(_sample):
        client = ScriptedModelClient(
            [ModelResponse(content='{"status":"unsupported","reason":"offline plumbing smoke"}')]
        )
        return LlmDecisionSource(client, InferenceSettings(model="offline-scripted-phase04-smoke"))

    summary = BatchRunner(
        dataset,
        decisions,
        BatchConfig(
            output=args.output,
            input_mode="text",
            limit=None if args.full else args.limit,
            resume=args.resume,
            workers=1,
            validate_official=True,
        ),
        transcript_provider=lambda sample: f"Phase 04 offline debug request for {sample.id}",
        run_identity={"model": "offline-scripted-phase04-smoke", "protocol": "phase04-v1"},
    ).run()
    print(
        json.dumps(
            {
                "selected_samples": summary.selected_samples,
                "completed_samples": summary.completed_samples,
                "failed_samples": summary.failed_samples,
                "ready_for_submission": summary.ready_for_submission,
                "output": str(summary.output) if summary.output else None,
                "metrics": str(summary.metrics),
                "failures": str(summary.failures),
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary.ready_for_submission else 1


if __name__ == "__main__":
    raise SystemExit(main())
