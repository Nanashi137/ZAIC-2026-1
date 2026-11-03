"""Command line entry point for text-debug or local-ASR public-test inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .agent import (
    AGENT_ARCHITECTURES,
    INVALID_RESPONSE_HISTORY_MODES,
    PROMPT_VERSIONS,
    LlmDecisionSource,
)
from .asr import (
    CachedTranscriber,
    ChunkFormerSettings,
    ChunkFormerTranscriber,
    Qwen3ASRSettings,
    Qwen3ASRTranscriber,
)
from .audio_pipeline import AudioInferencePipeline
from .batch import BatchConfig, BatchError, BatchRunner, load_transcript_file
from .compliance import (
    DEFAULT_SOFT_SAMPLE_TIMEOUT_SECONDS,
    HARD_SAMPLE_TIMEOUT_SECONDS,
    SampleDeadline,
    SampleDeadlineExceeded,
    configure_reproducibility,
)
from .data import DatasetReader
from .evaluation import (
    compare_evaluations,
    load_label_references,
    run_semantic_evaluation,
    select_sample_ids,
    write_comparison_markdown,
)
from .model import InferenceSettings, OpenAICompatibleClient, RetrySettings
from .runtime import ExplicitActionSource, Runtime, RuntimeConfig, ToolAction

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4.1-mini"
ASR_ENGINES = ("qwen", "chunkformer-rnnt", "chunkformer-ctc")


def _action(value: str) -> ToolAction:
    try:
        raw: Any = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"action is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("name"), str):
        raise argparse.ArgumentTypeError('action must be {"name": string, "arguments": object}')
    arguments = raw.get("arguments", {})
    if not isinstance(arguments, dict):
        raise argparse.ArgumentTypeError("action.arguments must be an object")
    return ToolAction(raw["name"], arguments)


def _arguments(value: str) -> dict[str, Any]:
    try:
        raw: Any = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"arguments are not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise argparse.ArgumentTypeError("arguments must be a JSON object")
    return raw


def _temperature(value: str) -> float | None:
    if value.lower() in {"none", "null", "omit"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("temperature must be a number or 'none'") from exc


def _add_asr_engine_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--asr-engine", choices=ASR_ENGINES, default="qwen")
    parser.add_argument("--asr-chunk-size", type=int, default=64)
    parser.add_argument("--asr-left-context-size", type=int, default=128)
    parser.add_argument("--asr-right-context-size", type=int, default=128)
    parser.add_argument("--asr-total-batch-duration", type=int, default=1800)
    parser.add_argument("--asr-rnnt-max-symbols-per-frame", type=int, default=2)


def _add_agent_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--agent-architecture", choices=AGENT_ARCHITECTURES, default="direct")
    parser.add_argument("--prompt-version", choices=PROMPT_VERSIONS, default="baseline")
    parser.add_argument(
        "--invalid-response-history",
        choices=INVALID_RESPONSE_HISTORY_MODES,
        default="full",
    )


def _build_base_asr_transcriber(args: argparse.Namespace, split_dir: Path):
    if args.asr_engine == "qwen":
        settings = Qwen3ASRSettings(
            checkpoint=args.asr_checkpoint,
            revision=args.asr_revision,
            backend=args.asr_backend,
            device=args.asr_device,
            dtype=args.asr_dtype,
            language=None if args.asr_language == "auto" else args.asr_language,
            max_output_tokens=args.asr_max_output_tokens,
            max_inference_batch_size=args.asr_max_batch_size,
        )
        return settings, Qwen3ASRTranscriber(settings, allowed_audio_root=split_dir)

    settings = ChunkFormerSettings(
        checkpoint=args.asr_checkpoint,
        revision=args.asr_revision,
        backend=args.asr_engine,
        device=args.asr_device,
        dtype=args.asr_dtype,
        language=None if args.asr_language == "auto" else "vi",
        chunk_size=args.asr_chunk_size,
        left_context_size=args.asr_left_context_size,
        right_context_size=args.asr_right_context_size,
        total_batch_duration=args.asr_total_batch_duration,
        rnnt_max_symbols_per_frame=(
            args.asr_rnnt_max_symbols_per_frame if args.asr_engine == "chunkformer-rnnt" else None
        ),
    )
    return settings, ChunkFormerTranscriber(settings, allowed_audio_root=split_dir)


def _asr_run_identity(settings: Any, args: argparse.Namespace) -> dict[str, Any]:
    identity = {
        **dict(settings.cache_identity()),
        "engine": args.asr_engine,
        "revision": settings.revision,
        "device": settings.device,
        "cache_enabled": not args.no_asr_cache,
        "cache_dir": str(args.asr_cache_dir.expanduser().resolve()),
    }
    if isinstance(settings, Qwen3ASRSettings):
        identity["max_inference_batch_size"] = settings.max_inference_batch_size
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/train+public_test"))
    parser.add_argument("--split", default="public_test", choices=("public_test",))
    parser.add_argument("--sample-id", required=True)
    parser.add_argument(
        "--input-mode",
        choices=("text", "audio"),
        default="text",
        help="Use an explicit debug transcript or transcribe official audio locally with Qwen3-ASR.",
    )
    parser.add_argument(
        "--transcript",
        help="Explicit local-debug text override; never a ground-truth public-test transcript.",
    )
    parser.add_argument(
        "--mode",
        choices=("explicit", "model"),
        default="explicit",
        help="Use deterministic explicit actions or the Phase 02 model loop.",
    )
    parser.add_argument(
        "--action",
        action="append",
        default=[],
        type=_action,
        help=(
            'Explicit deterministic action JSON; repeat for multiple calls. '
            'Example: {"name":"alarm.list","arguments":{}}'
        ),
    )
    parser.add_argument(
        "--tool",
        help="PowerShell-friendly single explicit tool name; appended after any --action values.",
    )
    parser.add_argument(
        "--arguments-json",
        type=_arguments,
        default={},
        help="Arguments for --tool as a JSON object (default: {}).",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--provider", choices=("openai-compatible",), default="openai-compatible")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=_temperature, default=0.0)
    parser.add_argument("--max-output-tokens", "--max-model-tokens", dest="max_output_tokens", type=int, default=1024)
    parser.add_argument("--timeout-seconds", "--timeout", dest="timeout_seconds", type=float, default=60.0)
    parser.add_argument(
        "--token-limit-field",
        choices=("auto", "max_tokens", "max_completion_tokens"),
        default="auto",
    )
    parser.add_argument("--tool-choice", choices=("auto", "required"), default="required")
    parser.add_argument("--requests-per-minute", type=float)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--retry-max-seconds", type=float, default=8.0)
    parser.add_argument("--max-invalid-proposals", type=int, default=2)
    parser.add_argument(
        "--agent-output-mode",
        choices=("native", "constrained-json"),
        default="native",
    )
    parser.add_argument("--competition-mode", action="store_true")
    parser.add_argument(
        "--sample-timeout-seconds", type=float, default=DEFAULT_SOFT_SAMPLE_TIMEOUT_SECONDS
    )
    parser.add_argument("--capture-raw-model-responses", action="store_true")
    _add_agent_experiment_arguments(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--asr-checkpoint", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--asr-backend",
        choices=("qwen-asr-transformers", "native-hf"),
        default="qwen-asr-transformers",
    )
    parser.add_argument("--asr-revision")
    parser.add_argument("--asr-device", default="auto", help="auto, cpu, cuda, or an explicit device such as cuda:0")
    parser.add_argument(
        "--asr-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto"
    )
    parser.add_argument(
        "--asr-language",
        choices=("Vietnamese", "auto"),
        default="Vietnamese",
        help="Force Vietnamese or allow Qwen3-ASR language detection.",
    )
    parser.add_argument("--asr-max-output-tokens", type=int, default=256)
    parser.add_argument("--asr-max-batch-size", type=int, default=1)
    parser.add_argument("--asr-cache-dir", type=Path, default=Path("cache/transcriptions"))
    parser.add_argument("--no-asr-cache", action="store_true")
    _add_asr_engine_arguments(parser)
    parser.add_argument(
        "--allow-external-data",
        action="store_true",
        help="Explicitly allow sending sample context/history/transcript to a non-local model endpoint.",
    )
    parser.add_argument("--submission-out", type=Path)
    parser.add_argument("--show-observations", action="store_true")
    return parser


def build_batch_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run resumable LifeMate batch inference.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/train+public_test"))
    parser.add_argument("--split", default="public_test", choices=("public_test",))
    parser.add_argument("--input-mode", choices=("audio", "text"), default="audio")
    parser.add_argument("--transcripts", type=Path, help="Debug JSONL rows with id/transcript for text batch mode.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-model-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=_temperature, default=0.0)
    parser.add_argument(
        "--token-limit-field",
        choices=("auto", "max_tokens", "max_completion_tokens"),
        default="auto",
    )
    parser.add_argument("--tool-choice", choices=("auto", "required"), default="required")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--requests-per-minute", type=float)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--retry-max-seconds", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--max-invalid-proposals", type=int, default=2)
    parser.add_argument(
        "--agent-output-mode",
        choices=("native", "constrained-json"),
        default="native",
    )
    parser.add_argument("--competition-mode", action="store_true")
    parser.add_argument(
        "--sample-timeout-seconds", type=float, default=DEFAULT_SOFT_SAMPLE_TIMEOUT_SECONDS
    )
    parser.add_argument("--capture-raw-model-responses", action="store_true")
    _add_agent_experiment_arguments(parser)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--asr-checkpoint", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--asr-backend", choices=("qwen-asr-transformers", "native-hf"), default="qwen-asr-transformers"
    )
    parser.add_argument("--asr-revision")
    parser.add_argument("--asr-device", default="auto")
    parser.add_argument("--asr-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--asr-language", choices=("Vietnamese", "auto"), default="Vietnamese")
    parser.add_argument("--asr-max-output-tokens", type=int, default=256)
    parser.add_argument("--asr-max-batch-size", type=int, default=1)
    parser.add_argument("--asr-cache-dir", type=Path, default=Path("cache/transcriptions"))
    parser.add_argument("--no-asr-cache", action="store_true")
    _add_asr_engine_arguments(parser)
    parser.add_argument("--allow-external-data", action="store_true")
    parser.add_argument("--skip-official-validation", action="store_true")
    return parser


def build_evaluate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run semantic evaluation on labeled train data.")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/train+public_test"))
    parser.add_argument("--split", default="train", choices=("train",))
    parser.add_argument("--input-mode", choices=("oracle", "asr"), default="oracle")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--sample-ids-file", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-model-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=_temperature, default=0.0)
    parser.add_argument(
        "--token-limit-field",
        choices=("auto", "max_tokens", "max_completion_tokens"),
        default="auto",
    )
    parser.add_argument("--tool-choice", choices=("auto", "required"), default="required")
    parser.add_argument("--requests-per-minute", type=float)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-base-seconds", type=float, default=1.0)
    parser.add_argument("--retry-max-seconds", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-tool-calls", type=int, default=6)
    parser.add_argument("--max-invalid-proposals", type=int, default=2)
    parser.add_argument(
        "--agent-output-mode",
        choices=("native", "constrained-json"),
        default="native",
    )
    parser.add_argument(
        "--compare-with-mode",
        choices=("native", "constrained-json"),
    )
    parser.add_argument("--competition-mode", action="store_true")
    parser.add_argument(
        "--sample-timeout-seconds", type=float, default=DEFAULT_SOFT_SAMPLE_TIMEOUT_SECONDS
    )
    parser.add_argument("--capture-raw-model-responses", action="store_true")
    _add_agent_experiment_arguments(parser)
    parser.add_argument("--no-deterministic", action="store_true")
    parser.add_argument("--asr-checkpoint", default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--asr-backend", choices=("qwen-asr-transformers", "native-hf"), default="qwen-asr-transformers"
    )
    parser.add_argument("--asr-revision")
    parser.add_argument("--asr-device", default="auto")
    parser.add_argument("--asr-dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--asr-language", choices=("Vietnamese", "auto"), default="Vietnamese")
    parser.add_argument("--asr-max-output-tokens", type=int, default=256)
    parser.add_argument("--asr-max-batch-size", type=int, default=1)
    parser.add_argument("--asr-cache-dir", type=Path, default=Path("cache/transcriptions"))
    parser.add_argument("--no-asr-cache", action="store_true")
    _add_asr_engine_arguments(parser)
    parser.add_argument("--allow-external-data", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if arguments and arguments[0] == "batch":
        return batch_main(arguments[1:])
    if arguments and arguments[0] == "evaluate":
        return evaluate_main(arguments[1:])
    parser = build_parser()
    args = parser.parse_args(arguments)
    if args.input_mode == "text" and not args.transcript:
        parser.error("--transcript is required with --input-mode text")
    if args.input_mode == "audio" and args.transcript:
        parser.error("--transcript cannot be combined with --input-mode audio")
    if args.sample_timeout_seconds is not None and args.sample_timeout_seconds <= 0:
        parser.error("--sample-timeout-seconds must be positive")
    if args.sample_timeout_seconds is not None and args.sample_timeout_seconds > HARD_SAMPLE_TIMEOUT_SECONDS:
        parser.error("--sample-timeout-seconds cannot exceed the 30-second hard sample budget")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.competition_mode and args.no_deterministic:
        parser.error("--competition-mode requires deterministic execution")
    if args.competition_mode and args.mode == "model" and args.temperature != 0.0:
        parser.error("--competition-mode requires --temperature 0")
    if args.mode == "model" and args.agent_architecture == "plan-ledger" and args.agent_output_mode != "native":
        parser.error("--agent-architecture plan-ledger requires --agent-output-mode native")
    configure_reproducibility(args.seed, deterministic=not args.no_deterministic)
    timeout = args.sample_timeout_seconds
    deadline = SampleDeadline.start(timeout) if timeout is not None else None
    actions = list(args.action)
    if args.tool:
        actions.append(ToolAction(args.tool, args.arguments_json))
    if args.mode == "model":
        if actions:
            parser.error("--action/--tool cannot be combined with --mode model")
        if not _is_local_url(args.base_url) and not args.allow_external_data:
            parser.error("a non-local --base-url requires explicit --allow-external-data approval")
        if args.competition_mode and not _is_local_url(args.base_url):
            parser.error("--competition-mode forbids non-local model endpoints")
        api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
        if _is_official_openai_url(args.base_url) and not api_key:
            parser.error(f"environment variable {args.api_key_env!r} is not set")
        client = OpenAICompatibleClient(
            base_url=args.base_url,
            api_key=api_key,
            timeout_seconds=args.timeout_seconds,
            retry_settings=RetrySettings(
                max_retries=args.max_retries,
                base_delay_seconds=args.retry_base_seconds,
                max_delay_seconds=args.retry_max_seconds,
            ),
            requests_per_minute=args.requests_per_minute,
        )
        decisions = LlmDecisionSource(
            client,
            InferenceSettings(
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                max_tokens_field=args.token_limit_field,
                tool_choice=args.tool_choice,
                agent_output_mode=args.agent_output_mode,
            ),
            max_invalid_proposals=args.max_invalid_proposals,
            deadline=deadline,
            capture_raw_responses=args.capture_raw_model_responses,
            agent_architecture=args.agent_architecture,
            prompt_version=args.prompt_version,
            invalid_response_history=args.invalid_response_history,
        )
    else:
        decisions = ExplicitActionSource(actions)
    dataset = DatasetReader(args.dataset_root, args.split)
    runtime = Runtime(
        dataset,
        config=RuntimeConfig(max_steps=args.max_steps, max_tool_calls=args.max_tool_calls),
    )
    audio_result = None
    try:
        if args.input_mode == "audio":
            _, transcriber = _build_base_asr_transcriber(args, dataset.split_dir)
            selected_transcriber = (
                transcriber if args.no_asr_cache else CachedTranscriber(transcriber, args.asr_cache_dir)
            )
            audio_result = AudioInferencePipeline(dataset, runtime, selected_transcriber).run(
                args.sample_id,
                decisions,
                deadline=deadline,
            )
            result = audio_result.run
        else:
            result = runtime.run(args.sample_id, args.transcript, decisions, deadline=deadline)
    except SampleDeadlineExceeded as exc:
        print(json.dumps({"termination_reason": "sample_timeout", "error": str(exc)}), file=sys.stderr)
        return 1
    record = dict(result.submission)
    print(json.dumps(record, ensure_ascii=False))
    diagnostics = {
        "decision_source": result.decision_source,
        "termination_reason": result.termination_reason,
        "tool_calls_sent": result.tool_calls_sent,
        "model_usage": dict(result.model_usage) if result.model_usage is not None else None,
        "model_latency_seconds": result.model_latency_seconds,
        "model_requests": [dict(item) for item in result.model_request_diagnostics],
        "phase_timings": dict(result.phase_timings),
        "steps": [
            {
                "index": step.index,
                "name": step.action.name if step.action is not None else None,
                "status": step.status,
                **({"error": step.error} if step.error else {}),
                **({"observation": dict(step.observation)} if args.show_observations and step.observation else {}),
            }
            for step in result.steps
        ],
    }
    if audio_result is not None:
        diagnostics["asr"] = {
            "checkpoint": audio_result.final_transcription.checkpoint,
            "backend": audio_result.final_transcription.backend,
            "detected_language": audio_result.final_transcription.language,
            "transcribed_turns": audio_result.diagnostics.transcribed_turns,
            "initialization_latency_seconds": (
                audio_result.diagnostics.initialization_latency_seconds
            ),
            "final_audio_latency_seconds": audio_result.diagnostics.final_audio_latency_seconds,
            "history_audio_latency_seconds": audio_result.diagnostics.history_audio_latency_seconds,
            "cache_hits": audio_result.diagnostics.cache_hits,
            "cache_hit_rate": audio_result.diagnostics.cache_hit_rate,
            "failure_count": audio_result.diagnostics.failure_count,
        }
    print(json.dumps(diagnostics, ensure_ascii=False), file=sys.stderr)
    if args.submission_out:
        args.submission_out.parent.mkdir(parents=True, exist_ok=True)
        args.submission_out.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


def batch_main(argv: Sequence[str] | None = None) -> int:
    parser = build_batch_parser()
    args = parser.parse_args(argv)
    if args.input_mode == "text" and args.transcripts is None:
        parser.error("--transcripts is required with batch --input-mode text")
    if args.input_mode == "audio" and args.transcripts is not None:
        parser.error("--transcripts is only valid with batch --input-mode text")
    if args.sample_timeout_seconds is not None and args.sample_timeout_seconds <= 0:
        parser.error("--sample-timeout-seconds must be positive")
    if args.sample_timeout_seconds is not None and args.sample_timeout_seconds > HARD_SAMPLE_TIMEOUT_SECONDS:
        parser.error("--sample-timeout-seconds cannot exceed the 30-second hard sample budget")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.competition_mode and args.no_deterministic:
        parser.error("--competition-mode requires deterministic execution")
    if args.competition_mode and args.temperature != 0.0:
        parser.error("--competition-mode requires --temperature 0")
    if args.agent_architecture == "plan-ledger" and args.agent_output_mode != "native":
        parser.error("--agent-architecture plan-ledger requires --agent-output-mode native")
    if not _is_local_url(args.base_url) and not args.allow_external_data:
        parser.error("a non-local --base-url requires explicit --allow-external-data approval")
    if args.competition_mode and not _is_local_url(args.base_url):
        parser.error("--competition-mode forbids non-local model endpoints")
    configure_reproducibility(args.seed, deterministic=not args.no_deterministic)

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if _is_official_openai_url(args.base_url) and not api_key:
        parser.error(f"environment variable {args.api_key_env!r} is not set")
    client = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key=api_key,
        timeout_seconds=args.timeout,
        retry_settings=RetrySettings(
            max_retries=args.max_retries,
            base_delay_seconds=args.retry_base_seconds,
            max_delay_seconds=args.retry_max_seconds,
        ),
        requests_per_minute=args.requests_per_minute,
    )
    inference_settings = InferenceSettings(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_model_tokens,
        max_tokens_field=args.token_limit_field,
        tool_choice=args.tool_choice,
        agent_output_mode=args.agent_output_mode,
    )

    def decisions(_sample):
        return LlmDecisionSource(
            client,
            inference_settings,
            max_invalid_proposals=args.max_invalid_proposals,
            capture_raw_responses=args.capture_raw_model_responses,
            agent_architecture=args.agent_architecture,
            prompt_version=args.prompt_version,
            invalid_response_history=args.invalid_response_history,
        )

    dataset = DatasetReader(args.dataset_root, args.split)
    runtime = Runtime(
        dataset,
        config=RuntimeConfig(max_steps=args.max_steps, max_tool_calls=args.max_tool_calls),
    )
    transcriber = None
    transcript_provider = None
    asr_identity: Mapping[str, Any] | None = None
    text_identity: Mapping[str, Any] | None = None
    if args.input_mode == "audio":
        asr_settings, base_transcriber = _build_base_asr_transcriber(args, dataset.split_dir)
        transcriber = (
            base_transcriber
            if args.no_asr_cache
            else CachedTranscriber(base_transcriber, args.asr_cache_dir)
        )
        asr_identity = _asr_run_identity(asr_settings, args)
    else:
        transcripts = load_transcript_file(args.transcripts)
        canonical_transcripts = json.dumps(
            transcripts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        text_identity = {
            "path": str(args.transcripts.expanduser().resolve()),
            "sha256": hashlib.sha256(canonical_transcripts).hexdigest(),
        }

        def transcript_provider(sample):
            try:
                return transcripts[sample.id]
            except KeyError as exc:
                raise ValueError(f"no debug transcript for {sample.id}") from exc

    run_identity = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": args.temperature,
        "max_model_tokens": args.max_model_tokens,
        "token_limit_field": args.token_limit_field,
        "tool_choice": args.tool_choice,
        "agent_output_mode": args.agent_output_mode,
        "agent_architecture": args.agent_architecture,
        "prompt_version": args.prompt_version,
        "invalid_response_history": args.invalid_response_history,
        "max_steps": args.max_steps,
        "max_tool_calls": args.max_tool_calls,
        "max_invalid_proposals": args.max_invalid_proposals,
        "max_retries": args.max_retries,
        "retry_base_seconds": args.retry_base_seconds,
        "retry_max_seconds": args.retry_max_seconds,
        "capture_raw_model_responses": args.capture_raw_model_responses,
        "competition_mode": args.competition_mode,
        "sample_timeout_seconds": (
            args.sample_timeout_seconds
            if args.sample_timeout_seconds is not None
            else DEFAULT_SOFT_SAMPLE_TIMEOUT_SECONDS
        ),
        "seed": args.seed,
        "deterministic": not args.no_deterministic,
        "asr": asr_identity,
        "text_transcripts": text_identity,
        "agent_protocol_version": "phase05-agent-experiment-v1",
    }
    try:
        summary = BatchRunner(
            dataset,
            decisions,
            BatchConfig(
                output=args.output,
                input_mode=args.input_mode,
                limit=args.limit,
                resume=args.resume,
                workers=args.workers,
                validate_official=not args.skip_official_validation,
                competition_mode=args.competition_mode,
                sample_timeout_seconds=args.sample_timeout_seconds,
                seed=args.seed,
                deterministic=not args.no_deterministic,
            ),
            runtime=runtime,
            transcriber=transcriber,
            transcript_provider=transcript_provider,
            run_identity=run_identity,
        ).run()
    except (BatchError, ValueError) as exc:
        print(f"batch failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "selected_samples": summary.selected_samples,
                "completed_samples": summary.completed_samples,
                "failed_samples": summary.failed_samples,
                "ready_for_submission": summary.ready_for_submission,
                "output": str(summary.output) if summary.output is not None else None,
                "metrics": str(summary.metrics),
                "failures": str(summary.failures),
            },
            ensure_ascii=False,
        )
    )
    return 0 if summary.ready_for_submission else 1


def evaluate_main(argv: Sequence[str] | None = None) -> int:
    parser = build_evaluate_parser()
    args = parser.parse_args(argv)
    if args.sample_timeout_seconds is not None and args.sample_timeout_seconds <= 0:
        parser.error("--sample-timeout-seconds must be positive")
    if args.sample_timeout_seconds is not None and args.sample_timeout_seconds > HARD_SAMPLE_TIMEOUT_SECONDS:
        parser.error("--sample-timeout-seconds cannot exceed the 30-second hard sample budget")
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.compare_with_mode == args.agent_output_mode:
        parser.error("--compare-with-mode must differ from --agent-output-mode")
    if args.agent_architecture == "plan-ledger" and (
        args.agent_output_mode != "native" or args.compare_with_mode is not None
    ):
        parser.error("plan-ledger evaluation requires native mode without --compare-with-mode")
    if args.competition_mode and args.no_deterministic:
        parser.error("--competition-mode requires deterministic execution")
    if args.competition_mode and args.temperature != 0.0:
        parser.error("--competition-mode requires --temperature 0")
    if not _is_local_url(args.base_url) and not args.allow_external_data:
        parser.error("a non-local --base-url requires explicit --allow-external-data approval")
    if args.competition_mode and not _is_local_url(args.base_url):
        parser.error("--competition-mode forbids non-local model endpoints")

    configure_reproducibility(args.seed, deterministic=not args.no_deterministic)
    dataset = DatasetReader(args.dataset_root, args.split, allow_train=True)
    references = load_label_references(dataset)

    selected_ids = _collect_selected_ids(args.sample_id, args.sample_ids_file)
    selected = select_sample_ids(references, sample_ids=selected_ids, limit=args.limit, seed=args.seed)
    runtime = Runtime(
        dataset,
        config=RuntimeConfig(max_steps=args.max_steps, max_tool_calls=args.max_tool_calls),
    )
    transcriber = None
    asr_settings = None
    if args.input_mode == "asr":
        asr_settings, base_transcriber = _build_base_asr_transcriber(args, dataset.split_dir)
        transcriber = base_transcriber if args.no_asr_cache else CachedTranscriber(base_transcriber, args.asr_cache_dir)

    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    if _is_official_openai_url(args.base_url) and not api_key:
        parser.error(f"environment variable {args.api_key_env!r} is not set")

    asr_identity = _asr_run_identity(asr_settings, args) if asr_settings is not None else None
    common_run_identity = {
        "model": args.model,
        "base_url": args.base_url,
        "input_mode": args.input_mode,
        "seed": args.seed,
        "deterministic": not args.no_deterministic,
        "competition_mode": args.competition_mode,
        "temperature": args.temperature,
        "max_model_tokens": args.max_model_tokens,
        "token_limit_field": args.token_limit_field,
        "tool_choice": args.tool_choice,
        "agent_architecture": args.agent_architecture,
        "prompt_version": args.prompt_version,
        "invalid_response_history": args.invalid_response_history,
        "max_steps": args.max_steps,
        "max_tool_calls": args.max_tool_calls,
        "max_invalid_proposals": args.max_invalid_proposals,
        "sample_timeout_seconds": args.sample_timeout_seconds,
        "model_timeout_seconds": args.timeout,
        "max_retries": args.max_retries,
        "retry_base_seconds": args.retry_base_seconds,
        "retry_max_seconds": args.retry_max_seconds,
        "requests_per_minute": args.requests_per_minute,
        "capture_raw_model_responses": args.capture_raw_model_responses,
        "asr": asr_identity,
    }
    requested_modes = [args.agent_output_mode]
    if args.compare_with_mode:
        requested_modes.append(args.compare_with_mode)
    evaluation_config = {
        "config_schema_version": 1,
        "dataset_root": str(dataset.dataset_root),
        "split": dataset.split,
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "selected_sample_count": len(selected),
        "selected_ids_sha256": hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest(),
        "requested_modes": requested_modes,
        **common_run_identity,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation-config.json").write_text(
        json.dumps(evaluation_config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    client = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key=api_key,
        timeout_seconds=args.timeout,
        retry_settings=RetrySettings(
            max_retries=args.max_retries,
            base_delay_seconds=args.retry_base_seconds,
            max_delay_seconds=args.retry_max_seconds,
        ),
        requests_per_minute=args.requests_per_minute,
    )

    reference_hash_cache: dict[tuple[str, str, str], tuple[str | None, str | None]] = {}

    def run_mode(mode: str, output_dir: Path):
        inference_settings = InferenceSettings(
            model=args.model,
            temperature=args.temperature,
            max_output_tokens=args.max_model_tokens,
            max_tokens_field=args.token_limit_field,
            tool_choice=args.tool_choice,
            agent_output_mode=mode,
        )

        def decision_factory(_sample):
            return LlmDecisionSource(
                client,
                inference_settings,
                max_invalid_proposals=args.max_invalid_proposals,
                capture_raw_responses=args.capture_raw_model_responses,
                agent_architecture=args.agent_architecture,
                prompt_version=args.prompt_version,
                invalid_response_history=args.invalid_response_history,
            )

        return run_semantic_evaluation(
            dataset=dataset,
            references=references,
            selected_ids=selected,
            runtime=runtime,
            decision_source_factory=decision_factory,
            output_dir=output_dir,
            input_mode=args.input_mode,
            sample_timeout_seconds=args.sample_timeout_seconds,
            transcriber=transcriber,
            reference_hash_cache=reference_hash_cache,
            run_identity={
                **common_run_identity,
                "agent_output_mode": mode,
            },
        )

    primary_dir = args.output_dir / f"{args.agent_output_mode}-{args.input_mode}"
    primary = run_mode(args.agent_output_mode, primary_dir)
    result: dict[str, Any] = {
        "mode": args.agent_output_mode,
        "records": str(primary.records_path),
        "metrics": str(primary.metrics_path),
        "report": str(primary.report_path),
        "sample_count": len(primary.selected_ids),
    }

    if args.compare_with_mode:
        secondary_dir = args.output_dir / f"{args.compare_with_mode}-{args.input_mode}"
        secondary = run_mode(args.compare_with_mode, secondary_dir)
        native, constrained = (
            (primary, secondary) if args.agent_output_mode == "native" else (secondary, primary)
        )
        comparison = compare_evaluations(native, constrained)
        comparison_path = args.output_dir / "comparison.json"
        comparison_markdown_path = args.output_dir / "comparison.md"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_comparison_markdown(comparison_markdown_path, comparison)
        result["compare_with_mode"] = args.compare_with_mode
        result["secondary_metrics"] = str(secondary.metrics_path)
        result["comparison_json"] = str(comparison_path)
        result["comparison_markdown"] = str(comparison_markdown_path)

    print(json.dumps(result, ensure_ascii=False))
    return 0


def _is_local_url(value: str) -> bool:
    host = (urlparse(value).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _is_official_openai_url(value: str) -> bool:
    return (urlparse(value).hostname or "").lower() == "api.openai.com"


def _collect_selected_ids(sample_ids: Sequence[str], sample_ids_file: Path | None) -> list[str]:
    values = [sample_id for sample_id in sample_ids if sample_id]
    if sample_ids_file is None:
        return values
    for line in sample_ids_file.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if item:
            values.append(item)
    return values


if __name__ == "__main__":
    raise SystemExit(main())
