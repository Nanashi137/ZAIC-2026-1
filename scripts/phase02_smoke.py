"""Offline scripted-model smoke through the real public-test ToolEnv.

This checks plumbing and official schema compatibility, not model quality or task success.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifemate.agent import LlmDecisionSource, ModelToolRegistry
from lifemate.data import DatasetReader
from lifemate.model import InferenceSettings, ModelResponse, ModelToolCall, ScriptedModelClient
from lifemate.runtime import Runtime


def _arguments(value: str) -> dict:
    raw = json.loads(value)
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError("arguments must be a JSON object")
    return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/train+public_test"))
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--arguments-json", type=_arguments, default={})
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    dataset = DatasetReader(args.dataset_root)
    sample = dataset.get_sample(args.sample_id)
    allowed = dataset.load_catalog().restricted_to(sample.allowed_tool_names)
    if args.tool not in allowed:
        parser.error(f"tool {args.tool!r} is not permitted for sample {args.sample_id!r}")
    registry = ModelToolRegistry.from_allowed(allowed)
    model_name = registry.official_to_model[args.tool]
    client = ScriptedModelClient(
        [
            ModelResponse(
                tool_calls=(ModelToolCall("offline_smoke_call", model_name, json.dumps(args.arguments_json)),)
            ),
            ModelResponse(content='{"status":"done","reason":"offline smoke complete"}'),
        ]
    )
    source = LlmDecisionSource(client, InferenceSettings(model="offline-scripted-smoke"))
    result = Runtime(dataset).run(args.sample_id, args.transcript, source)
    record = dict(result.submission)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False))
    print(
        json.dumps(
            {
                "decision_source": result.decision_source,
                "termination_reason": result.termination_reason,
                "tool_calls_sent": result.tool_calls_sent,
                "model_usage": dict(result.model_usage or {}),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
