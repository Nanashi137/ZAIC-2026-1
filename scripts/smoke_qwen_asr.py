"""Privacy-safe Qwen3-ASR smoke diagnostics for selected public-test audio.

This script prints transcript hashes and character counts, never transcript text.
It reads public input audio only and does not construct a ToolEnv or inspect state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lifemate.asr import Qwen3ASRSettings, Qwen3ASRTranscriber
from lifemate.data import DatasetReader


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/train+public_test"))
    parser.add_argument("--split", choices=("public_test",), default="public_test")
    parser.add_argument("--sample-id", default="public_0006")
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--language", choices=("Vietnamese", "auto"), default="Vietnamese")
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=2)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dataset = DatasetReader(args.dataset_root, args.split)
    sample = dataset.get_sample(args.sample_id)
    references: list[tuple[str, str]] = [("final", sample.audio)]
    references.extend(
        (f"history_user_{index}", str(turn["audio"]))
        for index, turn in enumerate(sample.history)
        if turn.get("role") == "user" and isinstance(turn.get("audio"), str)
    )
    references = references[: args.limit]

    settings = Qwen3ASRSettings(
        checkpoint=args.checkpoint,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        language=None if args.language == "auto" else args.language,
        max_output_tokens=args.max_output_tokens,
    )
    transcriber = Qwen3ASRTranscriber(settings, allowed_audio_root=dataset.split_dir)

    torch = _optional_torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    transcriber.initialize()
    initialization_memory = _cuda_memory(torch)

    results: list[dict[str, Any]] = []
    for role, reference in references:
        result = transcriber.transcribe(dataset.split_dir / reference)
        results.append(
            {
                "role": role,
                "detected_language": result.language,
                "latency_seconds": result.inference_latency_seconds,
                "characters": len(result.text),
                "text_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
            }
        )

    report = {
        "checkpoint": settings.checkpoint,
        "revision": settings.revision,
        "backend": settings.backend,
        "language_setting": settings.language,
        "device_setting": settings.device,
        "dtype_setting": settings.dtype,
        "python_packages": {
            name: _version(name) for name in ("qwen-asr", "torch", "transformers", "accelerate")
        },
        "gpu": torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else None,
        "initialization_latency_seconds": transcriber.initialization_latency_seconds,
        "initialization_cuda_memory_mib": initialization_memory,
        "final_cuda_memory_mib": _cuda_memory(torch),
        "files": results,
        "failure_count": 0,
    }
    print(json.dumps(report, ensure_ascii=False))
    return 0


def _optional_torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _cuda_memory(torch: Any) -> dict[str, float] | None:
    if torch is None or not torch.cuda.is_available():
        return None
    scale = 1024 * 1024
    return {
        "allocated": round(torch.cuda.memory_allocated() / scale, 1),
        "reserved": round(torch.cuda.memory_reserved() / scale, 1),
        "peak_allocated": round(torch.cuda.max_memory_allocated() / scale, 1),
    }


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
