"""Offline Phase 03 smoke: real cached Qwen transcripts, model loop, and official SDK."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lifemate.agent import LlmDecisionSource
from lifemate.asr import CachedTranscriber, Qwen3ASRSettings, Qwen3ASRTranscriber
from lifemate.audio_pipeline import AudioInferencePipeline
from lifemate.data import DatasetReader
from lifemate.model import InferenceSettings, ModelResponse, ScriptedModelClient
from lifemate.runtime import Runtime


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/train+public_test"))
    parser.add_argument("--sample-id", default="public_0006")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/transcriptions-smoke-20261101"))
    parser.add_argument("--output", type=Path, default=Path("outputs/phase03-smoke.jsonl"))
    args = parser.parse_args()

    dataset = DatasetReader(args.dataset_root)
    transcriber = CachedTranscriber(
        Qwen3ASRTranscriber(
            Qwen3ASRSettings(device="cuda", dtype="bfloat16", max_output_tokens=128),
            allowed_audio_root=dataset.split_dir,
        ),
        args.cache_dir,
    )
    client = ScriptedModelClient(
        [ModelResponse(content='{"status":"done","reason":"phase03 smoke complete"}')]
    )
    decisions = LlmDecisionSource(client, InferenceSettings(model="offline-scripted-phase03-smoke"))
    result = AudioInferencePipeline(dataset, Runtime(dataset), transcriber).run(args.sample_id, decisions)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(result.run.submission), ensure_ascii=False) + "\n", encoding="utf-8")
    diagnostics = {
        "decision_source": result.run.decision_source,
        "termination_reason": result.run.termination_reason,
        "model_requests": len(client.requests),
        "transcribed_turns": result.diagnostics.transcribed_turns,
        "cache_hits": result.diagnostics.cache_hits,
        "cache_hit_rate": result.diagnostics.cache_hit_rate,
        "tool_calls": len(result.run.submission["tool_calls"]),
        "state_hash_length": len(result.run.submission["state_hash"]),
    }
    print(json.dumps(diagnostics, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
