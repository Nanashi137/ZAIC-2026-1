"""Deterministic, resumable batch inference and diagnostic aggregation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

from tqdm.auto import tqdm

from .asr import Transcriber
from .audio_pipeline import AudioInferencePipeline
from .compliance import (
    DEFAULT_SOFT_SAMPLE_TIMEOUT_SECONDS,
    HARD_SAMPLE_TIMEOUT_SECONDS,
    SampleDeadline,
    SampleDeadlineExceeded,
    configure_reproducibility,
)
from .data import DatasetReader, Sample
from .runtime import DecisionSource, RunResult, Runtime
from .sdk import ToolSession


class BatchError(RuntimeError):
    """Base class for safe batch failures."""


class BatchResumeError(BatchError):
    """Existing checkpoint state is missing or incompatible."""


class SubmissionValidationError(BatchError):
    """A candidate submission is incomplete, duplicated, or malformed."""


DecisionSourceFactory = Callable[[Sample], DecisionSource]
TranscriptProvider = Callable[[Sample], str]
ProgressReporter = Callable[[int, int, str, str], None]


class OfficialValidator(Protocol):
    def validate(self, submission_path: Path) -> str: ...


@dataclass(frozen=True)
class BatchConfig:
    output: Path
    input_mode: str = "audio"
    limit: int | None = None
    resume: bool = False
    workers: int = 1
    validate_official: bool = True
    competition_mode: bool = False
    sample_timeout_seconds: float | None = DEFAULT_SOFT_SAMPLE_TIMEOUT_SECONDS
    seed: int = 42
    deterministic: bool = True

    def __post_init__(self) -> None:
        if self.input_mode not in {"audio", "text"}:
            raise ValueError("input_mode must be audio or text")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be positive")
        if self.workers < 1:
            raise ValueError("workers must be positive")
        if self.sample_timeout_seconds is not None and self.sample_timeout_seconds <= 0:
            raise ValueError("sample_timeout_seconds must be positive")
        if (
            self.sample_timeout_seconds is not None
            and self.sample_timeout_seconds > HARD_SAMPLE_TIMEOUT_SECONDS
        ):
            raise ValueError("sample_timeout_seconds cannot exceed the hard sample budget")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.competition_mode:
            object.__setattr__(self, "validate_official", False)
            if self.sample_timeout_seconds is None:
                object.__setattr__(
                    self, "sample_timeout_seconds", DEFAULT_SOFT_SAMPLE_TIMEOUT_SECONDS
                )


@dataclass(frozen=True)
class BatchPaths:
    output: Path
    progress: Path
    metrics: Path
    failures: Path
    manifest: Path

    @classmethod
    def from_output(cls, output: Path | str) -> BatchPaths:
        output_path = Path(output).expanduser().resolve()
        directory = output_path.parent
        return cls(
            output=output_path,
            progress=directory / "batch-progress.jsonl",
            metrics=directory / "metrics.json",
            failures=directory / "failures.jsonl",
            manifest=directory / "batch-manifest.json",
        )


@dataclass(frozen=True)
class BatchSampleRecord:
    id: str
    index: int
    submission: Mapping[str, Any]
    termination_reason: str
    elapsed_seconds: float
    model_latency_seconds: float
    asr_inference_seconds: float
    asr_initialization_seconds: float
    asr_cache_hits: int
    transcribed_turns: int
    model_requests: int
    input_tokens: int
    output_tokens: int
    total_tokens: int
    step_statuses: tuple[str, ...]
    model_request_diagnostics: tuple[Mapping[str, Any], ...] = ()
    phase_timings: Mapping[str, float] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["submission"] = dict(self.submission)
        return value

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> BatchSampleRecord:
        return cls(
            id=str(value["id"]),
            index=int(value["index"]),
            submission=dict(value["submission"]),
            termination_reason=str(value["termination_reason"]),
            elapsed_seconds=float(value["elapsed_seconds"]),
            model_latency_seconds=float(value["model_latency_seconds"]),
            asr_inference_seconds=float(value["asr_inference_seconds"]),
            asr_initialization_seconds=float(value.get("asr_initialization_seconds", 0.0)),
            asr_cache_hits=int(value["asr_cache_hits"]),
            transcribed_turns=int(value["transcribed_turns"]),
            model_requests=int(value["model_requests"]),
            input_tokens=int(value["input_tokens"]),
            output_tokens=int(value["output_tokens"]),
            total_tokens=int(value["total_tokens"]),
            step_statuses=tuple(str(item) for item in value["step_statuses"]),
            model_request_diagnostics=tuple(
                dict(item)
                for item in value.get("model_request_diagnostics", [])
                if isinstance(item, Mapping)
            ),
            phase_timings={
                str(key): float(item)
                for key, item in dict(value.get("phase_timings", {})).items()
            },
        )


@dataclass(frozen=True)
class BatchFailureRecord:
    id: str
    index: int
    error_type: str
    message: str
    elapsed_seconds: float
    submission: Mapping[str, Any] | None = None
    termination_reason: str | None = None
    step_statuses: tuple[str, ...] = ()
    model_latency_seconds: float = 0.0
    asr_inference_seconds: float = 0.0
    asr_initialization_seconds: float = 0.0
    asr_cache_hits: int = 0
    transcribed_turns: int = 0
    model_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    tool_calls_sent: int = 0
    model_request_diagnostics: tuple[Mapping[str, Any], ...] = ()
    step_diagnostics: tuple[Mapping[str, Any], ...] = ()
    phase_timings: Mapping[str, float] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        value = asdict(self)
        value["submission"] = dict(self.submission) if self.submission is not None else None
        return value

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> BatchFailureRecord:
        return cls(
            id=str(value["id"]),
            index=int(value["index"]),
            error_type=str(value["error_type"]),
            message=str(value["message"]),
            elapsed_seconds=float(value["elapsed_seconds"]),
            submission=(
                dict(value["submission"])
                if isinstance(value.get("submission"), Mapping)
                else None
            ),
            termination_reason=(
                str(value["termination_reason"]) if value.get("termination_reason") is not None else None
            ),
            step_statuses=tuple(str(item) for item in value.get("step_statuses", [])),
            model_latency_seconds=float(value.get("model_latency_seconds", 0.0)),
            asr_inference_seconds=float(value.get("asr_inference_seconds", 0.0)),
            asr_initialization_seconds=float(value.get("asr_initialization_seconds", 0.0)),
            asr_cache_hits=int(value.get("asr_cache_hits", 0)),
            transcribed_turns=int(value.get("transcribed_turns", 0)),
            model_requests=int(value.get("model_requests", 0)),
            input_tokens=int(value.get("input_tokens", 0)),
            output_tokens=int(value.get("output_tokens", 0)),
            total_tokens=int(value.get("total_tokens", 0)),
            tool_calls_sent=int(value.get("tool_calls_sent", 0)),
            model_request_diagnostics=tuple(
                dict(item)
                for item in value.get("model_request_diagnostics", [])
                if isinstance(item, Mapping)
            ),
            step_diagnostics=tuple(
                dict(item)
                for item in value.get("step_diagnostics", [])
                if isinstance(item, Mapping)
            ),
            phase_timings={
                str(key): float(item)
                for key, item in dict(value.get("phase_timings", {})).items()
            },
        )


@dataclass(frozen=True)
class BatchRunSummary:
    selected_samples: int
    completed_samples: int
    failed_samples: int
    ready_for_submission: bool
    output: Path | None
    metrics: Path
    failures: Path


class SubprocessOfficialValidator:
    """Run the installed split-local official schema validator in a separate process."""

    def __init__(self, dataset: DatasetReader) -> None:
        self.dataset = dataset

    def validate(self, submission_path: Path) -> str:
        command = [
            sys.executable,
            "-m",
            "zaic_tools",
            "--split",
            self.dataset.split,
            "--root",
            str(self.dataset.dataset_root),
            "--validate",
            str(submission_path),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=self.dataset.split_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise SubmissionValidationError(f"could not run official validator: {exc}") from exc
        output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
        if result.returncode != 0:
            raise SubmissionValidationError(
                f"official validator failed with exit code {result.returncode}: {output[:1000]}"
            )
        return output


class _LockedTranscriber:
    """Serialize access to one reusable model when batch workers exceed one."""

    def __init__(self, inner: Transcriber) -> None:
        self.inner = inner
        self.settings = inner.settings
        self._lock = threading.Lock()

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        deadline: SampleDeadline | None = None,
    ):
        with self._lock:
            return self.inner.transcribe(audio_path, deadline=deadline)


class BatchRunner:
    def __init__(
        self,
        dataset: DatasetReader,
        decision_source_factory: DecisionSourceFactory,
        config: BatchConfig,
        *,
        runtime: Runtime | None = None,
        transcriber: Transcriber | None = None,
        transcript_provider: TranscriptProvider | None = None,
        run_identity: Mapping[str, Any] | None = None,
        official_validator: OfficialValidator | None = None,
        progress_reporter: ProgressReporter | None = None,
        clock: Callable[[], float] = time.perf_counter,
        deadline_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.dataset = dataset
        self.decision_source_factory = decision_source_factory
        self.config = config
        if config.competition_mode:
            configure_reproducibility(config.seed, deterministic=config.deterministic)
        self.runtime = runtime or Runtime(dataset)
        if self.runtime.dataset.dataset_root != dataset.dataset_root or self.runtime.dataset.split != dataset.split:
            raise ValueError("batch runner and runtime must use the same dataset split")
        if config.input_mode == "audio" and transcriber is None:
            raise ValueError("audio batch mode requires a transcriber")
        if config.input_mode == "text" and transcript_provider is None:
            raise ValueError("text batch mode requires a transcript provider")
        self.transcriber = (
            _LockedTranscriber(transcriber) if transcriber is not None and config.workers > 1 else transcriber
        )
        self.transcript_provider = transcript_provider
        self.paths = BatchPaths.from_output(config.output)
        self.run_identity = dict(run_identity or {})
        self.official_validator = official_validator or SubprocessOfficialValidator(dataset)
        self.progress_reporter = progress_reporter
        self.clock = clock
        self.deadline_clock = deadline_clock
        self._sample_indexes: dict[str, int] = {}

    def run(self) -> BatchRunSummary:
        samples = list(self.dataset.iter_samples())
        dataset_count = len(samples)
        if self.config.limit is not None:
            samples = samples[: self.config.limit]
        self._sample_indexes = {sample.id: index for index, sample in enumerate(samples)}
        expected_ids = [sample.id for sample in samples]
        if len(expected_ids) != len(set(expected_ids)):
            raise SubmissionValidationError("selected input contains duplicate sample IDs")
        manifest = self._manifest(dataset_count, expected_ids)
        completed, failures = self._prepare(manifest, expected_ids)
        pending = [sample for sample in samples if sample.id not in completed]
        total = len(samples)
        processed = len(completed)
        progress_bar = None
        progress_reporter = self.progress_reporter
        if progress_reporter is None:
            description_parts = [str(self.run_identity.get("model", "batch"))]
            asr_identity = self.run_identity.get("asr")
            if isinstance(asr_identity, Mapping) and asr_identity.get("engine"):
                description_parts.append(str(asr_identity["engine"]))
            description_parts.append(self.config.input_mode)
            progress_bar = tqdm(
                total=total,
                initial=processed,
                desc=" ".join(description_parts),
                unit="sample",
                dynamic_ncols=True,
            )

            def progress_reporter(completed_count: int, _total: int, sample_id: str, status: str) -> None:
                assert progress_bar is not None
                progress_bar.set_postfix_str(f"{sample_id}: {status}", refresh=False)
                progress_bar.update(max(0, completed_count - progress_bar.n))

        iterator = map(self._process_sample, pending)
        executor: ThreadPoolExecutor | None = None
        if self.config.workers > 1 and pending:
            executor = ThreadPoolExecutor(max_workers=self.config.workers)
            iterator = executor.map(self._process_sample, pending)
        try:
            for outcome in iterator:
                processed += 1
                if isinstance(outcome, BatchSampleRecord):
                    completed[outcome.id] = outcome
                    failures.pop(outcome.id, None)
                    status = outcome.termination_reason
                else:
                    failures[outcome.id] = outcome
                    status = f"failed:{outcome.error_type}"
                self._checkpoint(samples, completed, failures, dataset_count)
                progress_reporter(processed, total, outcome.id, status)
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            if progress_bar is not None:
                progress_bar.close()

        ordered_records = [completed[sample.id] for sample in samples if sample.id in completed]
        rows_by_id = {
            sample_id: dict(record.submission) for sample_id, record in completed.items()
        }
        rows_by_id.update(
            {
                sample_id: dict(failure.submission)
                for sample_id, failure in failures.items()
                if failure.submission is not None
            }
        )
        rows = [rows_by_id[sample.id] for sample in samples if sample.id in rows_by_id]
        ready = len(rows) == total
        official_result: str | None = None
        output: Path | None = None
        if ready:
            validate_submission_records(rows, expected_ids)
            candidate = self.paths.output.with_name(f".{self.paths.output.name}.candidate")
            _write_jsonl(candidate, rows)
            try:
                if self.config.validate_official:
                    official_result = self.official_validator.validate(candidate)
                os.replace(candidate, self.paths.output)
            finally:
                candidate.unlink(missing_ok=True)
            output = self.paths.output
        else:
            self.paths.output.unlink(missing_ok=True)

        metrics = build_metrics(dataset_count, total, ordered_records, failures, ready, official_result)
        _write_json(self.paths.metrics, metrics)
        return BatchRunSummary(
            total,
            len(ordered_records),
            len(failures),
            ready,
            output,
            self.paths.metrics,
            self.paths.failures,
        )

    def _process_sample(self, sample: Sample) -> BatchSampleRecord | BatchFailureRecord:
        started = self.clock()
        active_session: ToolSession | None = None

        def remember_session(session: ToolSession) -> None:
            nonlocal active_session
            active_session = session

        deadline = (
            SampleDeadline.start(self.config.sample_timeout_seconds, clock=self.deadline_clock)
            if self.config.sample_timeout_seconds is not None
            else None
        )
        try:
            decisions = self.decision_source_factory(sample)
            if deadline is not None:
                deadline.check("agent initialization")
                set_deadline = getattr(decisions, "set_deadline", None)
                if callable(set_deadline):
                    set_deadline(deadline)
            if self.config.input_mode == "audio":
                assert self.transcriber is not None
                audio_result = AudioInferencePipeline(self.dataset, self.runtime, self.transcriber).run(
                    sample.id,
                    decisions,
                    deadline=deadline,
                    session_observer=remember_session,
                )
                run = audio_result.run
                asr_seconds = (
                    audio_result.diagnostics.final_audio_latency_seconds
                    + audio_result.diagnostics.history_audio_latency_seconds
                )
                asr_initialization_seconds = (
                    audio_result.diagnostics.initialization_latency_seconds
                )
                cache_hits = audio_result.diagnostics.cache_hits
                transcribed_turns = audio_result.diagnostics.transcribed_turns
            else:
                assert self.transcript_provider is not None
                transcript = self.transcript_provider(sample)
                if not transcript.strip():
                    raise ValueError(f"no text transcript supplied for {sample.id}")
                run = self.runtime.run(
                    sample.id,
                    transcript,
                    decisions,
                    deadline=deadline,
                    session_observer=remember_session,
                )
                asr_seconds = 0.0
                asr_initialization_seconds = 0.0
                cache_hits = 0
                transcribed_turns = 0
            if run.termination_reason in _FAILED_TERMINATIONS:
                return _failure_from_run(
                    sample,
                    self._sample_indexes[sample.id],
                    run,
                    self.clock() - started,
                    asr_seconds,
                    asr_initialization_seconds,
                    cache_hits,
                    transcribed_turns,
                )
            return _sample_record(
                sample,
                self._sample_indexes[sample.id],
                run,
                self.clock() - started,
                asr_seconds,
                asr_initialization_seconds,
                cache_hits,
                transcribed_turns,
            )
        except Exception as exc:
            termination = "sample_timeout" if isinstance(exc, SampleDeadlineExceeded) else None
            submission: Mapping[str, Any] | None = None
            fallback_error: Exception | None = None
            try:
                submission = dict(
                    self.runtime.capture_submission(
                        sample.id,
                        session=active_session,
                    )
                )
            except Exception as capture_exc:
                fallback_error = capture_exc
            message = str(exc)[:1000]
            if fallback_error is not None:
                fallback_detail = (
                    f"; submission fallback failed: {type(fallback_error).__name__}: "
                    f"{fallback_error}"
                )
                message = (message + fallback_detail)[:1000]
            return BatchFailureRecord(
                id=sample.id,
                index=self._sample_indexes.get(sample.id, -1),
                error_type=type(exc).__name__,
                message=message,
                elapsed_seconds=self.clock() - started,
                submission=submission,
                termination_reason=termination,
                tool_calls_sent=(
                    len(submission["tool_calls"])
                    if submission is not None
                    and isinstance(submission.get("tool_calls"), list)
                    else 0
                ),
            )

    def _prepare(
        self, manifest: Mapping[str, Any], expected_ids: Sequence[str]
    ) -> tuple[dict[str, BatchSampleRecord], dict[str, BatchFailureRecord]]:
        self.paths.output.parent.mkdir(parents=True, exist_ok=True)
        state_paths = (
            self.paths.output,
            self.paths.progress,
            self.paths.metrics,
            self.paths.failures,
            self.paths.manifest,
        )
        existing = [path for path in state_paths if path.exists()]
        if not self.config.resume and existing:
            names = ", ".join(path.name for path in existing)
            raise BatchResumeError(f"batch output state already exists ({names}); use --resume or a new directory")
        if not existing:
            _write_json(self.paths.manifest, manifest)
            _write_jsonl(self.paths.progress, [])
            _write_jsonl(self.paths.failures, [])
            return {}, {}
        if not self.config.resume:
            raise BatchResumeError("existing batch state requires --resume")
        if not self.paths.manifest.is_file() or not self.paths.progress.is_file():
            raise BatchResumeError("resume requires batch-manifest.json and batch-progress.jsonl")
        existing_manifest = json.loads(self.paths.manifest.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise BatchResumeError("resume configuration or selected sample set does not match the checkpoint")
        completed: dict[str, BatchSampleRecord] = {}
        for value in _read_jsonl(self.paths.progress):
            record = BatchSampleRecord.from_json(value)
            if record.id in completed:
                raise BatchResumeError(f"duplicate completed sample in checkpoint: {record.id}")
            if record.index < 0 or record.index >= len(expected_ids) or expected_ids[record.index] != record.id:
                raise BatchResumeError(f"completed checkpoint index mismatch for {record.id}")
            try:
                validate_submission_records([record.submission], [record.id])
            except SubmissionValidationError as exc:
                raise BatchResumeError(f"invalid completed checkpoint for {record.id}: {exc}") from exc
            completed[record.id] = record
        failures: dict[str, BatchFailureRecord] = {}
        if self.paths.failures.is_file():
            for value in _read_jsonl(self.paths.failures):
                failure = BatchFailureRecord.from_json(value)
                if failure.id in failures:
                    raise BatchResumeError(f"duplicate failed sample in checkpoint: {failure.id}")
                if failure.index < 0 or failure.index >= len(expected_ids) or expected_ids[failure.index] != failure.id:
                    raise BatchResumeError(f"failed checkpoint index mismatch for {failure.id}")
                if failure.submission is not None:
                    try:
                        validate_submission_records([failure.submission], [failure.id])
                    except SubmissionValidationError as exc:
                        raise BatchResumeError(
                            f"invalid degraded checkpoint for {failure.id}: {exc}"
                        ) from exc
                failures[failure.id] = failure
        unknown = (set(completed) | set(failures)) - set(expected_ids)
        if unknown:
            raise BatchResumeError(f"checkpoint contains unknown sample IDs: {sorted(unknown)[:5]}")
        return completed, failures

    def _checkpoint(
        self,
        samples: Sequence[Sample],
        completed: Mapping[str, BatchSampleRecord],
        failures: Mapping[str, BatchFailureRecord],
        dataset_count: int,
    ) -> None:
        ordered_records = [completed[sample.id] for sample in samples if sample.id in completed]
        ordered_failures = [failures[sample.id] for sample in samples if sample.id in failures]
        _write_jsonl(self.paths.progress, [record.as_json() for record in ordered_records])
        _write_jsonl(self.paths.failures, [record.as_json() for record in ordered_failures])
        metrics = build_metrics(
            dataset_count,
            len(samples),
            ordered_records,
            failures,
            False,
            None,
        )
        _write_json(self.paths.metrics, metrics)

    def _manifest(self, dataset_count: int, expected_ids: Sequence[str]) -> dict[str, Any]:
        ids_digest = hashlib.sha256("\n".join(expected_ids).encode("utf-8")).hexdigest()
        return {
            "schema_version": 1,
            "dataset_root": str(self.dataset.dataset_root),
            "split": self.dataset.split,
            "dataset_samples": dataset_count,
            "selected_samples": len(expected_ids),
            "selected_ids_sha256": ids_digest,
            "input_mode": self.config.input_mode,
            "run_identity": self.run_identity,
        }


def _sample_record(
    sample: Sample,
    index: int,
    run: RunResult,
    elapsed_seconds: float,
    asr_seconds: float,
    asr_initialization_seconds: float,
    cache_hits: int,
    transcribed_turns: int,
) -> BatchSampleRecord:
    usage = dict(run.model_usage or {})
    phase_timings = _batch_phase_timings(
        run, elapsed_seconds, asr_seconds, asr_initialization_seconds
    )
    return BatchSampleRecord(
        id=sample.id,
        index=index,
        submission=dict(run.submission),
        termination_reason=run.termination_reason,
        elapsed_seconds=elapsed_seconds,
        model_latency_seconds=run.model_latency_seconds or 0.0,
        asr_inference_seconds=asr_seconds,
        asr_initialization_seconds=asr_initialization_seconds,
        asr_cache_hits=cache_hits,
        transcribed_turns=transcribed_turns,
        model_requests=int(usage.get("requests", 0)),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        total_tokens=int(usage.get("total_tokens", 0)),
        step_statuses=tuple(step.status for step in run.steps),
        model_request_diagnostics=tuple(dict(item) for item in run.model_request_diagnostics),
        phase_timings=phase_timings,
    )


def _failure_from_run(
    sample: Sample,
    index: int,
    run: RunResult,
    elapsed_seconds: float,
    asr_seconds: float,
    asr_initialization_seconds: float,
    cache_hits: int,
    transcribed_turns: int,
) -> BatchFailureRecord:
    usage = dict(run.model_usage or {})
    phase_timings = _batch_phase_timings(
        run, elapsed_seconds, asr_seconds, asr_initialization_seconds
    )
    step_errors = [step.error for step in run.steps if step.error]
    detail = f": {step_errors[-1]}" if step_errors else ""
    return BatchFailureRecord(
        id=sample.id,
        index=index,
        error_type="AgentTerminationFailure",
        message=f"agent terminated without a complete result: {run.termination_reason}{detail}",
        elapsed_seconds=elapsed_seconds,
        submission=dict(run.submission),
        termination_reason=run.termination_reason,
        step_statuses=tuple(step.status for step in run.steps),
        model_latency_seconds=run.model_latency_seconds or 0.0,
        asr_inference_seconds=asr_seconds,
        asr_initialization_seconds=asr_initialization_seconds,
        asr_cache_hits=cache_hits,
        transcribed_turns=transcribed_turns,
        model_requests=int(usage.get("requests", 0)),
        input_tokens=int(usage.get("input_tokens", 0)),
        output_tokens=int(usage.get("output_tokens", 0)),
        total_tokens=int(usage.get("total_tokens", 0)),
        tool_calls_sent=run.tool_calls_sent,
        model_request_diagnostics=tuple(dict(item) for item in run.model_request_diagnostics),
        step_diagnostics=_step_diagnostics(run),
        phase_timings=phase_timings,
    )


def _batch_phase_timings(
    run: RunResult,
    elapsed_seconds: float,
    asr_seconds: float,
    asr_initialization_seconds: float,
) -> dict[str, float]:
    timings = {str(key): float(value) for key, value in run.phase_timings.items()}
    timings["asr_inference_seconds"] = asr_seconds
    timings["asr_initialization_seconds"] = asr_initialization_seconds
    timings["sample_wall_seconds"] = elapsed_seconds
    accounted = (
        asr_initialization_seconds
        + asr_seconds
        + float(timings.get("agent_total_seconds", 0.0))
    )
    timings["pipeline_overhead_seconds"] = max(0.0, elapsed_seconds - accounted)
    return timings


def _step_diagnostics(run: RunResult) -> tuple[Mapping[str, Any], ...]:
    local_statuses = {
        "model_bad_output",
        "model_bad_output_limit",
        "permission_rejected",
        "schema_rejected",
        "invalid_finish",
        "agent_finished",
        "duplicate_write_rejected",
        "duplicate_read_rejected",
        "tool_call_budget_rejected",
        "sample_timeout",
    }
    return tuple(
        {
            "step_index": step.index,
            "selected_tool": step.action.name if step.action is not None else None,
            "status": step.status,
            "sdk_invoked": step.action is not None and step.status not in local_statuses,
            "validation_ok": step.validation.get("ok") if step.validation is not None else None,
            "error": step.error,
        }
        for step in run.steps
    )


def validate_submission_records(rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]) -> None:
    ids: list[str] = []
    for index, row in enumerate(rows):
        if set(row) != {"id", "tool_calls", "state_hash"}:
            raise SubmissionValidationError(f"row {index} must contain exactly id/tool_calls/state_hash")
        sample_id = row.get("id")
        calls = row.get("tool_calls")
        state_hash = row.get("state_hash")
        if not isinstance(sample_id, str):
            raise SubmissionValidationError(f"row {index} id must be a string")
        if not isinstance(calls, list):
            raise SubmissionValidationError(f"row {index} tool_calls must be a list")
        for call_index, call in enumerate(calls):
            if (
                not isinstance(call, Mapping)
                or set(call) != {"name", "arguments"}
                or not isinstance(call.get("name"), str)
                or not isinstance(call.get("arguments"), Mapping)
            ):
                raise SubmissionValidationError(f"row {index} tool call {call_index} is malformed")
        if not isinstance(state_hash, str) or not state_hash:
            raise SubmissionValidationError(f"row {index} state_hash must be non-empty")
        ids.append(sample_id)
    if len(ids) != len(set(ids)):
        raise SubmissionValidationError("submission contains duplicate IDs")
    if ids != list(expected_ids):
        missing = sorted(set(expected_ids) - set(ids))
        extra = sorted(set(ids) - set(expected_ids))
        raise SubmissionValidationError(
            f"submission coverage/order mismatch; missing={missing[:5]}, extra={extra[:5]}"
        )


def build_metrics(
    dataset_samples: int,
    selected_samples: int,
    records: Sequence[BatchSampleRecord],
    failures: Mapping[str, BatchFailureRecord],
    ready: bool,
    official_validation: str | None,
) -> dict[str, Any]:
    all_outcomes: list[BatchSampleRecord | BatchFailureRecord] = [*records, *failures.values()]
    statuses = Counter(status for record in all_outcomes for status in record.step_statuses)
    terminations = Counter(record.termination_reason for record in records)
    terminations.update(
        failure.termination_reason or f"failed:{failure.error_type}" for failure in failures.values()
    )
    latencies = [record.elapsed_seconds for record in all_outcomes]
    total_requests = sum(record.model_requests for record in all_outcomes)
    return {
        "diagnostic_only": True,
        "official_accuracy_score": None,
        "dataset_samples": dataset_samples,
        "total_samples": selected_samples,
        "completed_samples": len(records),
        "failed_samples": len(failures),
        "degraded_submission_rows": sum(
            failure.submission is not None for failure in failures.values()
        ),
        "submission_coverage_rows": len(records)
        + sum(failure.submission is not None for failure in failures.values()),
        "samples_with_tool_calls": (
            sum(bool(record.submission["tool_calls"]) for record in records)
            + sum(failure.tool_calls_sent > 0 for failure in failures.values())
        ),
        "invalid_model_proposals": statuses["model_bad_output"] + statuses["model_bad_output_limit"],
        "schema_validation_errors": statuses["schema_rejected"],
        "sdk_tool_errors": statuses["sdk_rejected"] + statuses["tool_error"],
        "uncertain_executions": statuses["execution_uncertain"],
        "no_op_tool_calls": statuses["no_op"],
        "duplicate_writes_rejected": statuses["duplicate_write_rejected"],
        "duplicate_reads_rejected": statuses["duplicate_read_rejected"],
        "tool_budget_proposals_rejected": statuses["tool_call_budget_rejected"],
        "clarification_requests": terminations["awaiting_user"],
        "unsupported_requests": terminations["unsupported_request"],
        "termination_reasons": dict(sorted(terminations.items())),
        "latency_seconds": _latency_metrics(latencies),
        "asr_inference_seconds": {
            "total": sum(record.asr_inference_seconds for record in all_outcomes),
            "average_per_processed": _average([record.asr_inference_seconds for record in all_outcomes]),
        },
        "asr_initialization_seconds": {
            "total": sum(record.asr_initialization_seconds for record in all_outcomes),
            "average_per_processed": _average(
                [record.asr_initialization_seconds for record in all_outcomes]
            ),
        },
        "asr_transcribed_turns": sum(record.transcribed_turns for record in all_outcomes),
        "asr_cache_hits": sum(record.asr_cache_hits for record in all_outcomes),
        "llm_inference_seconds": {
            "total": sum(record.model_latency_seconds for record in all_outcomes),
            "average_per_processed": _average([record.model_latency_seconds for record in all_outcomes]),
        },
        "llm_requests": {
            "total": total_requests,
            "average_per_processed": total_requests / len(all_outcomes) if all_outcomes else 0.0,
        },
        "token_usage": {
            "input_tokens": sum(record.input_tokens for record in all_outcomes),
            "output_tokens": sum(record.output_tokens for record in all_outcomes),
            "total_tokens": sum(record.total_tokens for record in all_outcomes),
        },
        "ready_for_submission": ready,
        "official_schema_validation": official_validation,
    }


def load_transcript_file(path: Path | str) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    for value in _read_jsonl(Path(path)):
        sample_id = value.get("id")
        transcript = value.get("transcript")
        if not isinstance(sample_id, str) or not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("each transcript row requires non-empty id and transcript strings")
        if sample_id in transcripts:
            raise ValueError(f"duplicate transcript ID: {sample_id}")
        transcripts[sample_id] = transcript
    return transcripts


def _latency_metrics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"average": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(values)
    return {
        "average": sum(ordered) / len(ordered),
        "p50": _percentile(ordered, 50),
        "p90": _percentile(ordered, 90),
        "p95": _percentile(ordered, 95),
        "p99": _percentile(ordered, 99),
    }


def _percentile(ordered: Sequence[float], percentile: int) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    if not path.is_file():
        return values
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BatchResumeError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise BatchResumeError(f"{path}:{line_number}: expected object")
            values.append(value)
    return values


def _write_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n" for value in values)
    _atomic_write(path, text)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
            handle.write(text)
            temporary = handle.name
        os.replace(temporary, path)
    except OSError:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


_FAILED_TERMINATIONS = frozenset(
    {
        "model_bad_output_limit",
        "model_refusal",
        "model_error",
        "max_steps_exhausted",
        "tool_call_budget_exhausted",
        "duplicate_write_blocked",
        "duplicate_recovery_exhausted",
        "read_loop_exhausted",
        "execution_uncertain",
        "sdk_rejected",
        "tool_error",
        "sample_timeout",
    }
)
