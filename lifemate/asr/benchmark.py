"""Standalone, resumable WER/CER/latency benchmark for local ASR backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .chunkformer import (
    CHUNKFORMER_CTC_BACKEND,
    CHUNKFORMER_RNNT_BACKEND,
    ChunkFormerSettings,
    ChunkFormerTranscriber,
)
from .common import Transcriber, audio_duration_seconds, sha256_file
from .phowhisper import PhoWhisperSettings, PhoWhisperTranscriber
from .qwen import Qwen3ASRSettings, Qwen3ASRTranscriber
from .whisper import (
    FasterWhisperSettings,
    FasterWhisperTranscriber,
    WhisperSettings,
    WhisperTranscriber,
)

BENCHMARK_SCHEMA_VERSION = 2
_CHUNKFORMER_BACKENDS = {"chunkformer", CHUNKFORMER_RNNT_BACKEND, CHUNKFORMER_CTC_BACKEND}
_CHUNKFORMER_RNNT_BACKENDS = {"chunkformer", CHUNKFORMER_RNNT_BACKEND}


@dataclass(frozen=True)
class BenchmarkSample:
    sample_id: str
    audio_path: Path
    audio_reference: str
    reference: str
    duration_seconds: float
    audio_sha256: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark one local ASR checkpoint on fixed labeled-train final clips (no agent runtime)."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/train+public_test"))
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--backend",
        choices=(
            "qwen",
            "phowhisper",
            "whisper",
            "faster-whisper",
            "chunkformer",
            CHUNKFORMER_RNNT_BACKEND,
            CHUNKFORMER_CTC_BACKEND,
        ),
        required=True,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--parameter-count-checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--language", choices=("vi", "auto"), default="vi")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--max-audio-seconds", type=float, default=30.0)
    parser.add_argument("--warmup-samples", type=int, default=1)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--qwen-backend",
        choices=("qwen-asr-transformers", "native-hf"),
        default="qwen-asr-transformers",
    )
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--left-context-size", type=int, default=128)
    parser.add_argument("--right-context-size", type=int, default=128)
    parser.add_argument("--total-batch-duration", type=int, default=1800)
    parser.add_argument("--rnnt-max-symbols-per-frame", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.split != "train":
        parser.error("the standalone benchmark requires the labeled train split")
    if args.limit < 1 or args.seed < 0:
        parser.error("--limit must be positive and --seed must be non-negative")
    if args.max_audio_seconds <= 0 or args.warmup_samples < 0:
        parser.error("audio limit must be positive and warmup count must be non-negative")
    if args.max_output_tokens < 1:
        parser.error("--max-output-tokens must be positive")
    if args.backend in _CHUNKFORMER_RNNT_BACKENDS and args.rnnt_max_symbols_per_frame < 1:
        parser.error("--rnnt-max-symbols-per-frame must be positive")
    try:
        result = run_benchmark(args)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(1, f"ASR benchmark failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["complete"] else 2


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.expanduser().resolve()
    split_dir = dataset_root / args.split
    selected_ids = _requested_ids(args.sample_id, args.sample_ids_file)
    samples = load_samples(
        split_dir=split_dir,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        requested_ids=selected_ids,
    )
    _enforce_duration_limit(samples, args.max_audio_seconds)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.jsonl"
    failures_path = output_dir / "failures.jsonl"
    config_path = output_dir / "config.json"
    records_path = output_dir / "records.jsonl"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"

    run_spec = _run_spec(args, dataset_root, samples)
    fingerprint = _fingerprint(run_spec)
    config = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        **run_spec,
    }
    _prepare_config(config_path, config, resume=args.resume)
    completed = _load_checkpoint(checkpoint_path, samples, fingerprint) if args.resume else {}
    pending = [sample for sample in samples if sample.sample_id not in completed]
    previous_metrics = _load_json_object(metrics_path) if metrics_path.exists() else {}

    transcriber: Transcriber | None = None
    initialization_latency = _nonnegative_float(previous_metrics.get("initialization_latency_seconds"))
    parameter_count = _nonnegative_int(previous_metrics.get("parameter_count"))
    parameter_count_source = _nonempty_string(previous_metrics.get("parameter_count_source"))
    needs_initialization = bool(pending) or parameter_count is None or parameter_count_source is None
    if needs_initialization:
        _reset_cuda_peak_memory(args.device)
        transcriber = _build_transcriber(args, split_dir)
        initializer = getattr(transcriber, "initialize", None)
        if callable(initializer):
            initializer()
        initialization_latency = _nonnegative_float(
            getattr(transcriber, "initialization_latency_seconds", None)
        )
        parameter_count = _nonnegative_int(getattr(transcriber, "parameter_count", None))
        parameter_count_source = _nonempty_string(
            getattr(transcriber, "parameter_count_source", None)
        )
        if parameter_count is None or parameter_count_source is None:
            raise RuntimeError("the selected backend did not expose an exact parameter count and source")
        if pending:
            for sample in samples[: args.warmup_samples]:
                transcriber.transcribe(sample.audio_path)

    failures: list[dict[str, Any]] = []
    try:
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("tqdm is required; install the asr-benchmark dependency group") from exc

    with checkpoint_path.open("a", encoding="utf-8") as checkpoint_handle, failures_path.open(
        "a", encoding="utf-8"
    ) as failure_handle:
        progress = tqdm(pending, total=len(samples), initial=len(completed), unit="clip", desc=args.backend)
        for sample in progress:
            assert transcriber is not None
            try:
                _cuda_synchronize(args.device)
                started = time.perf_counter()
                result = transcriber.transcribe(sample.audio_path)
                _cuda_synchronize(args.device)
                wall_latency = time.perf_counter() - started
                row = _record(sample, result, wall_latency, fingerprint)
                _append_jsonl(checkpoint_handle, row)
                completed[sample.sample_id] = row
                progress.set_postfix(wer=f"{row['normalized']['word_error_rate']:.3f}")
            except Exception as exc:  # keep other clips/checkpoints usable after one decoder failure
                failure = {
                    "sample_id": sample.sample_id,
                    "run_fingerprint": fingerprint,
                    "attempted_at_unix": time.time(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                _append_jsonl(failure_handle, failure)
                failures.append(failure)

    ordered_records = [completed[sample.sample_id] for sample in samples if sample.sample_id in completed]
    _write_jsonl_atomic(records_path, ordered_records)
    gpu = (
        _gpu_metrics(args.device)
        if needs_initialization
        else _mapping_or_default(previous_metrics.get("gpu"))
    )
    metrics = aggregate_metrics(
        samples=samples,
        records=ordered_records,
        failures=failures,
        run_fingerprint=fingerprint,
        initialization_latency_seconds=initialization_latency,
        parameter_count=parameter_count,
        parameter_count_source=parameter_count_source,
        gpu=gpu,
    )
    _write_json_atomic(metrics_path, metrics)
    _write_text_atomic(report_path, _render_report(config, metrics))
    return {
        "complete": metrics["complete"],
        "completed_samples": metrics["completed_samples"],
        "failed_samples_this_attempt": len(failures),
        "metrics_path": str(metrics_path),
        "records_path": str(records_path),
        "resume_checkpoint": str(checkpoint_path),
    }


def load_samples(
    *,
    split_dir: Path,
    split: str,
    limit: int,
    seed: int,
    requested_ids: Sequence[str],
) -> list[BenchmarkSample]:
    input_path = split_dir / f"{split}.jsonl"
    raw_by_id: dict[str, tuple[str, str]] = {}
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise ValueError(f"{input_path}:{line_number}: expected an object")
            sample_id = raw.get("id")
            audio = raw.get("audio")
            reference = raw.get("audio_text")
            if not all(isinstance(value, str) and value.strip() for value in (sample_id, audio, reference)):
                raise ValueError(f"{input_path}:{line_number}: id/audio/audio_text must be non-empty strings")
            if sample_id in raw_by_id:
                raise ValueError(f"{input_path}:{line_number}: duplicate id {sample_id!r}")
            raw_by_id[sample_id] = (audio, reference)

    if requested_ids:
        ids = []
        seen: set[str] = set()
        for sample_id in requested_ids:
            if sample_id not in raw_by_id:
                raise KeyError(f"unknown train sample id {sample_id!r}")
            if sample_id not in seen:
                ids.append(sample_id)
                seen.add(sample_id)
        ids = ids[:limit]
    else:
        ids = list(raw_by_id)
        random.Random(seed).shuffle(ids)
        ids = ids[:limit]

    samples: list[BenchmarkSample] = []
    root = split_dir.resolve()
    for sample_id in ids:
        audio_reference, reference = raw_by_id[sample_id]
        audio_path = (root / audio_reference).resolve()
        if not audio_path.is_relative_to(root):
            raise ValueError(f"audio path escapes train split: {audio_reference!r}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")
        samples.append(
            BenchmarkSample(
                sample_id=sample_id,
                audio_path=audio_path,
                audio_reference=audio_reference,
                reference=reference,
                duration_seconds=audio_duration_seconds(audio_path),
                audio_sha256=sha256_file(audio_path),
            )
        )
    return samples


def _enforce_duration_limit(samples: Sequence[BenchmarkSample], limit: float) -> None:
    too_long = [sample for sample in samples if sample.duration_seconds > limit]
    if not too_long:
        return
    examples = ", ".join(f"{sample.sample_id}={sample.duration_seconds:.3f}s" for sample in too_long[:5])
    raise ValueError(
        f"{len(too_long)} selected clips exceed the {limit:.3f}s no-chunking limit ({examples}); "
        "change the benchmark protocol explicitly instead of silently chunking"
    )


def _build_transcriber(args: argparse.Namespace, split_dir: Path) -> Transcriber:
    language = None if args.language == "auto" else "vi"
    if args.backend == "qwen":
        return Qwen3ASRTranscriber(
            Qwen3ASRSettings(
                checkpoint=args.checkpoint,
                revision=args.revision,
                backend=args.qwen_backend,
                device=args.device,
                dtype=args.dtype,
                language=None if language is None else "Vietnamese",
                max_output_tokens=args.max_output_tokens,
            ),
            allowed_audio_root=split_dir,
        )
    if args.backend == "phowhisper":
        return PhoWhisperTranscriber(
            PhoWhisperSettings(
                checkpoint=args.checkpoint,
                revision=args.revision,
                device=args.device,
                dtype=args.dtype,
                language=language,
                max_output_tokens=args.max_output_tokens,
                max_audio_seconds=args.max_audio_seconds,
            ),
            allowed_audio_root=split_dir,
        )
    if args.backend == "whisper":
        return WhisperTranscriber(
            WhisperSettings(
                checkpoint=args.checkpoint,
                revision=args.revision,
                device=args.device,
                dtype=args.dtype,
                language=language,
                max_output_tokens=args.max_output_tokens,
                max_audio_seconds=args.max_audio_seconds,
            ),
            allowed_audio_root=split_dir,
        )
    if args.backend == "faster-whisper":
        return FasterWhisperTranscriber(
            FasterWhisperSettings(
                checkpoint=args.checkpoint,
                revision=args.revision,
                parameter_count_checkpoint=args.parameter_count_checkpoint,
                device=args.device,
                dtype=args.dtype,
                language=language,
                max_output_tokens=args.max_output_tokens,
                max_audio_seconds=args.max_audio_seconds,
            ),
            allowed_audio_root=split_dir,
        )
    chunkformer_backend = (
        CHUNKFORMER_CTC_BACKEND
        if args.backend == CHUNKFORMER_CTC_BACKEND
        else CHUNKFORMER_RNNT_BACKEND
    )
    return ChunkFormerTranscriber(
        ChunkFormerSettings(
            checkpoint=args.checkpoint,
            revision=args.revision,
            backend=chunkformer_backend,
            device=args.device,
            dtype=args.dtype,
            language=language,
            chunk_size=args.chunk_size,
            left_context_size=args.left_context_size,
            right_context_size=args.right_context_size,
            total_batch_duration=args.total_batch_duration,
            rnnt_max_symbols_per_frame=(
                args.rnnt_max_symbols_per_frame
                if chunkformer_backend == CHUNKFORMER_RNNT_BACKEND
                else None
            ),
        ),
        allowed_audio_root=split_dir,
    )


def _record(sample: BenchmarkSample, result: Any, wall_latency: float, fingerprint: str) -> dict[str, Any]:
    raw = _error_counts(_collapse_whitespace(sample.reference), _collapse_whitespace(result.text))
    normalized_reference = normalize_text(sample.reference)
    normalized_hypothesis = normalize_text(result.text)
    normalized = _error_counts(normalized_reference, normalized_hypothesis)
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_fingerprint": fingerprint,
        "sample_id": sample.sample_id,
        "audio_reference": sample.audio_reference,
        "audio_sha256": sample.audio_sha256,
        "audio_duration_seconds": sample.duration_seconds,
        "reference": sample.reference,
        "hypothesis": result.text,
        "normalized_reference": normalized_reference,
        "normalized_hypothesis": normalized_hypothesis,
        "language": result.language,
        "checkpoint": result.checkpoint,
        "backend": result.backend,
        "backend_inference_latency_seconds": result.inference_latency_seconds,
        "wall_latency_seconds": wall_latency,
        "real_time_factor": wall_latency / sample.duration_seconds if sample.duration_seconds else None,
        "raw": raw,
        "normalized": normalized,
    }


def normalize_text(text: str) -> str:
    """Conservative shared normalization: NFC, lowercase, punctuation to spaces."""

    normalized = unicodedata.normalize("NFC", text).lower()
    normalized = "".join(" " if unicodedata.category(char)[0] in {"P", "S"} else char for char in normalized)
    return _collapse_whitespace(normalized)


def _error_counts(reference: str, hypothesis: str) -> dict[str, Any]:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    word_errors = levenshtein_distance(reference_words, hypothesis_words)
    reference_chars = list(reference.replace(" ", ""))
    hypothesis_chars = list(hypothesis.replace(" ", ""))
    char_errors = levenshtein_distance(reference_chars, hypothesis_chars)
    return {
        "word_errors": word_errors,
        "reference_words": len(reference_words),
        "word_error_rate": word_errors / len(reference_words) if reference_words else float(bool(hypothesis_words)),
        "char_errors": char_errors,
        "reference_chars": len(reference_chars),
        "char_error_rate": char_errors / len(reference_chars) if reference_chars else float(bool(hypothesis_chars)),
        "exact_match": reference == hypothesis,
    }


def levenshtein_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_item in enumerate(left, 1):
        current = [left_index]
        for right_index, right_item in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def aggregate_metrics(
    *,
    samples: Sequence[BenchmarkSample],
    records: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    run_fingerprint: str,
    initialization_latency_seconds: float | None,
    parameter_count: int | None,
    parameter_count_source: str | None,
    gpu: Mapping[str, Any],
) -> dict[str, Any]:
    duration = sum(float(record["audio_duration_seconds"]) for record in records)
    latencies = [float(record["wall_latency_seconds"]) for record in records]
    result: dict[str, Any] = {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "selected_samples": len(samples),
        "completed_samples": len(records),
        "complete": len(records) == len(samples),
        "failed_samples_this_attempt": len(failures),
        "initialization_latency_seconds": initialization_latency_seconds,
        "parameter_count": parameter_count,
        "parameter_count_source": parameter_count_source,
        "audio_duration_seconds": duration,
        "latency_seconds": {
            "total": sum(latencies),
            "mean": statistics.fmean(latencies) if latencies else None,
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
        },
        "real_time_factor": sum(latencies) / duration if duration else None,
        "gpu": dict(gpu),
    }
    for field in ("raw", "normalized"):
        word_errors = sum(int(record[field]["word_errors"]) for record in records)
        reference_words = sum(int(record[field]["reference_words"]) for record in records)
        char_errors = sum(int(record[field]["char_errors"]) for record in records)
        reference_chars = sum(int(record[field]["reference_chars"]) for record in records)
        result[field] = {
            "word_errors": word_errors,
            "reference_words": reference_words,
            "word_error_rate": word_errors / reference_words if reference_words else None,
            "char_errors": char_errors,
            "reference_chars": reference_chars,
            "char_error_rate": char_errors / reference_chars if reference_chars else None,
            "exact_matches": sum(bool(record[field]["exact_match"]) for record in records),
        }
    return result


def _run_spec(args: argparse.Namespace, dataset_root: Path, samples: Sequence[BenchmarkSample]) -> dict[str, Any]:
    return {
        "benchmark_schema_version": BENCHMARK_SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "split": args.split,
        "scope": "final_user_audio_only",
        "backend_family": args.backend,
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "device": args.device,
        "dtype": args.dtype,
        "language": args.language,
        "qwen_backend": args.qwen_backend if args.backend == "qwen" else None,
        "parameter_count_checkpoint": (
            args.parameter_count_checkpoint if args.backend == "faster-whisper" else None
        ),
        "max_output_tokens": args.max_output_tokens,
        "max_audio_seconds": args.max_audio_seconds,
        "no_chunking": args.backend in {"phowhisper", "whisper", "faster-whisper"},
        "chunk_size": args.chunk_size if args.backend in _CHUNKFORMER_BACKENDS else None,
        "left_context_size": args.left_context_size if args.backend in _CHUNKFORMER_BACKENDS else None,
        "right_context_size": args.right_context_size if args.backend in _CHUNKFORMER_BACKENDS else None,
        "total_batch_duration": (
            args.total_batch_duration if args.backend in _CHUNKFORMER_BACKENDS else None
        ),
        "rnnt_max_symbols_per_frame": (
            args.rnnt_max_symbols_per_frame
            if args.backend in _CHUNKFORMER_RNNT_BACKENDS
            else None
        ),
        "warmup_samples": args.warmup_samples,
        "seed": args.seed,
        "selected_sample_count": len(samples),
        "selected_ids": [sample.sample_id for sample in samples],
        "selected_ids_sha256": hashlib.sha256(
            "\n".join(sample.sample_id for sample in samples).encode("utf-8")
        ).hexdigest(),
        "selected_audio_sha256": hashlib.sha256(
            "\n".join(sample.audio_sha256 for sample in samples).encode("ascii")
        ).hexdigest(),
        "selected_duration_seconds": sum(sample.duration_seconds for sample in samples),
        "max_selected_duration_seconds": max((sample.duration_seconds for sample in samples), default=0.0),
        "text_normalization": "NFC + lowercase + Unicode punctuation/symbol to space + collapse whitespace",
    }


def _load_checkpoint(
    path: Path,
    samples: Sequence[BenchmarkSample],
    fingerprint: str,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    expected = {sample.sample_id: sample for sample in samples}
    completed: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: corrupt checkpoint JSON") from exc
            sample_id = row.get("sample_id")
            if row.get("run_fingerprint") != fingerprint:
                raise ValueError(f"{path}:{line_number}: checkpoint configuration mismatch")
            if sample_id not in expected:
                raise ValueError(f"{path}:{line_number}: checkpoint contains an unselected sample")
            if sample_id in completed:
                raise ValueError(f"{path}:{line_number}: duplicate checkpoint sample {sample_id!r}")
            sample = expected[sample_id]
            if row.get("audio_sha256") != sample.audio_sha256 or row.get("reference") != sample.reference:
                raise ValueError(f"{path}:{line_number}: source data changed for {sample_id}")
            completed[sample_id] = row
    return completed


def _prepare_config(path: Path, config: Mapping[str, Any], *, resume: bool) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if not resume:
            raise ValueError(f"output directory already contains {path.name}; choose a new directory or use --resume")
        if existing.get("run_fingerprint") != config["run_fingerprint"]:
            raise ValueError("existing output config differs from this run; choose a new output directory")
        return
    _write_json_atomic(path, config)


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _requested_ids(cli_ids: Sequence[str], ids_file: Path | None) -> list[str]:
    values = list(cli_ids)
    if ids_file is not None:
        values.extend(line.strip() for line in ids_file.read_text(encoding="utf-8-sig").splitlines() if line.strip())
    return values


def _append_jsonl(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _write_text_atomic(path, "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows))


def _write_text_atomic(path: Path, content: str) -> None:
    temporary: str | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = handle.name
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _fingerprint(value: Mapping[str, Any]) -> str:
    canonical = json.dumps(dict(value), ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _collapse_whitespace(text: str) -> str:
    return " ".join(text.split())


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _cuda_synchronize(device: str) -> None:
    if not str(device).startswith("cuda") and device != "auto":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        return


def _reset_cuda_peak_memory(device: str) -> None:
    if not str(device).startswith("cuda") and device != "auto":
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        return


def _gpu_metrics(device: str) -> dict[str, Any]:
    if not str(device).startswith("cuda") and device != "auto":
        return {"available": False}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False}
        index = torch.cuda.current_device()
        return {
            "available": True,
            "name": torch.cuda.get_device_name(index),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(index),
        }
    except (ImportError, RuntimeError):
        return {"available": False}


def _render_report(config: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    normalized = metrics["normalized"]
    latency = metrics["latency_seconds"]
    lines = [
        "# ASR benchmark report",
        "",
        f"- Backend: `{config['backend_family']}`",
        f"- Checkpoint: `{config['checkpoint']}`",
        f"- Final clips: {metrics['completed_samples']}/{metrics['selected_samples']}",
        f"- Complete: {metrics['complete']}",
        f"- Normalized WER: {_format_metric(normalized['word_error_rate'])}",
        f"- Normalized CER: {_format_metric(normalized['char_error_rate'])}",
        f"- Exact matches: {normalized['exact_matches']}",
        f"- Total measured latency: {_format_metric(latency['total'])} s",
        f"- Mean / p50 / p95 latency: {_format_metric(latency['mean'])} / "
        f"{_format_metric(latency['p50'])} / {_format_metric(latency['p95'])} s",
        f"- Real-time factor: {_format_metric(metrics['real_time_factor'])}",
        f"- Exact parameter count: {_format_integer(metrics['parameter_count'])}",
        f"- Parameter count source: `{metrics['parameter_count_source']}`",
        f"- Max selected duration: {_format_metric(config['max_selected_duration_seconds'])} s",
        f"- Duration guard: {config['max_audio_seconds']} s; no silent Whisper-family chunking",
    ]
    if config["backend_family"] in _CHUNKFORMER_RNNT_BACKENDS:
        lines.append(
            f"- RNN-T max symbols per acoustic frame: {config['rnnt_max_symbols_per_frame']}"
        )
    elif config["backend_family"] == CHUNKFORMER_CTC_BACKEND:
        lines.append("- Decoder: CTC greedy")
    lines.extend(
        [
            "",
            "WER/CER use the same conservative normalization for every backend. See `config.json` for the exact "
            "selection hash and settings; `checkpoint.jsonl` is append-only and safe to resume.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.6f}"


def _nonnegative_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
        return float(value)
    return None


def _nonnegative_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _nonempty_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _format_integer(value: Any) -> str:
    return "n/a" if value is None else f"{int(value):,}"


def _mapping_or_default(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {"available": False}


if __name__ == "__main__":
    sys.exit(main())
