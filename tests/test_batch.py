from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from lifemate.agent import LlmDecisionSource
from lifemate.asr import Qwen3ASRSettings, TranscriptionResult
from lifemate.batch import (
    BatchConfig,
    BatchRunner,
    SubmissionValidationError,
    validate_submission_records,
)
from lifemate.data import DatasetReader
from lifemate.model import InferenceSettings, ModelResponse, ModelUsage, ScriptedModelClient
from lifemate.runtime import ExplicitActionSource, Runtime, ToolAction
from tests.support import FakeSession


def _batch_dataset(tmp_path: Path, count: int = 3, *, audio: bool = False) -> DatasetReader:
    split = tmp_path / "public_test"
    split.mkdir()
    rows: list[dict[str, Any]] = []
    for index in range(count):
        sample_id = f"public_{index:04d}"
        reference = f"audio/{sample_id}/{sample_id}.wav"
        rows.append(
            {
                "id": sample_id,
                "audio": reference,
                "context": {"installed_apps": {"alarm": {"tools": ["alarm.list"]}}},
                "history": [],
            }
        )
        if audio:
            path = split / reference
            path.parent.mkdir(parents=True)
            path.write_bytes(b"fixture")
    (split / "public_test.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    catalog = {
        "schema_version": "test",
        "apps": [
            {
                "id": "fixture",
                "tools": [
                    {"name": "alarm.list", "kind": "read", "parameters": {"type": "object", "properties": {}}},
                    {
                        "name": "sys.ask_missing_info",
                        "kind": "meta",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "tool_name": {"type": "string"},
                                "params": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["tool_name", "params"],
                        },
                    },
                ],
            }
        ],
    }
    (split / "public_test_tools_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    return DatasetReader(tmp_path)


class PerSampleFactory:
    def __init__(self, *, fail_once: str | None = None) -> None:
        self.fail_once = fail_once
        self.creations: list[str] = []

    def create(self, split: str, sample_id: str, dataset_root: Path) -> FakeSession:
        del split, dataset_root
        self.creations.append(sample_id)
        if self.fail_once == sample_id:
            self.fail_once = None
            raise RuntimeError("fixture interruption")
        return FakeSession(
            {"alarm.list", "sys.ask_missing_info"},
            state_hash=f"sdk-hash-{sample_id}",
        )


class RecordingValidator:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def validate(self, submission_path: Path) -> str:
        self.rows = [json.loads(line) for line in submission_path.read_text(encoding="utf-8").splitlines()]
        return "fixture official validation passed"


class AlwaysDoneClient:
    name = "always-done"

    def __init__(self) -> None:
        self.requests = []

    def complete(self, request):
        self.requests.append(request)
        return ModelResponse(
            content='{"status":"done","reason":"fixture"}',
            usage=ModelUsage(10, 2, 12),
        )


class FakeTranscriber:
    settings = Qwen3ASRSettings(checkpoint="fixture/Qwen3-ASR")

    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe(self, audio_path: Path | str, *, deadline=None) -> TranscriptionResult:
        del deadline
        name = Path(audio_path).name
        self.calls.append(name)
        return TranscriptionResult(
            text=f"request {name}",
            language="Vietnamese",
            inference_latency_seconds=0.1,
            audio_sha256=name,
            checkpoint=self.settings.checkpoint,
            backend=self.settings.backend,
        )


def test_batch_preserves_order_isolates_sessions_and_resets_model_conversation(tmp_path: Path) -> None:
    dataset = _batch_dataset(tmp_path)
    session_factory = PerSampleFactory()
    client = AlwaysDoneClient()
    validator = RecordingValidator()

    def decisions(_sample):
        return LlmDecisionSource(client, InferenceSettings(model="fixture"))

    summary = BatchRunner(
        dataset,
        decisions,
        BatchConfig(output=tmp_path / "out" / "submission.jsonl", input_mode="text", workers=2),
        runtime=Runtime(dataset, session_factory=session_factory),
        transcript_provider=lambda sample: f"debug {sample.id}",
        run_identity={"model": "fixture"},
        official_validator=validator,
        progress_reporter=lambda *_args: None,
    ).run()

    expected = [f"public_{index:04d}" for index in range(3)]
    assert summary.ready_for_submission
    assert [row["id"] for row in validator.rows] == expected
    assert [row["state_hash"] for row in validator.rows] == [f"sdk-hash-{item}" for item in expected]
    assert sorted(session_factory.creations) == expected
    assert len(session_factory.creations) == 3
    assert len(client.requests) == 3
    assert all(len(request.messages) == 2 for request in client.requests)
    metrics = json.loads(summary.metrics.read_text(encoding="utf-8"))
    assert metrics["completed_samples"] == 3
    assert metrics["llm_requests"]["total"] == 3
    assert metrics["official_accuracy_score"] is None


def test_resume_skips_completed_samples_and_retries_only_failure(tmp_path: Path) -> None:
    dataset = _batch_dataset(tmp_path)
    factory = PerSampleFactory(fail_once="public_0001")
    output = tmp_path / "resume" / "submission.jsonl"
    common = {
        "dataset": dataset,
        "decision_source_factory": lambda _sample: ExplicitActionSource([]),
        "runtime": Runtime(dataset, session_factory=factory),
        "transcript_provider": lambda sample: f"debug {sample.id}",
        "run_identity": {"fixture": True},
        "official_validator": RecordingValidator(),
        "progress_reporter": lambda *_args: None,
    }

    first = BatchRunner(
        config=BatchConfig(output=output, input_mode="text", validate_official=False),
        **common,
    ).run()

    assert not first.ready_for_submission
    assert not output.exists()
    assert first.completed_samples == 2
    assert first.failed_samples == 1
    assert factory.creations == ["public_0000", "public_0001", "public_0002"]

    second = BatchRunner(
        config=BatchConfig(output=output, input_mode="text", resume=True, validate_official=False),
        **common,
    ).run()

    assert second.ready_for_submission
    assert factory.creations == ["public_0000", "public_0001", "public_0002", "public_0001"]
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["public_0000", "public_0001", "public_0002"]
    assert (output.parent / "failures.jsonl").read_text(encoding="utf-8") == ""


def test_audio_batch_reuses_one_transcriber_and_outputs_sdk_calls(tmp_path: Path) -> None:
    dataset = _batch_dataset(tmp_path, count=2, audio=True)
    factory = PerSampleFactory()
    transcriber = FakeTranscriber()
    output = tmp_path / "audio-out" / "submission.jsonl"

    summary = BatchRunner(
        dataset,
        lambda _sample: ExplicitActionSource([ToolAction("alarm.list", {})]),
        BatchConfig(output=output, input_mode="audio", validate_official=False),
        runtime=Runtime(dataset, session_factory=factory),
        transcriber=transcriber,
        run_identity={"asr": "fixture"},
        progress_reporter=lambda *_args: None,
    ).run()

    assert summary.ready_for_submission
    assert transcriber.calls == ["public_0000.wav", "public_0001.wav"]
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert all(row["tool_calls"] == [{"name": "alarm.list", "arguments": {}}] for row in rows)


def test_submission_validation_rejects_missing_duplicate_and_malformed_rows() -> None:
    valid = {"id": "a", "tool_calls": [], "state_hash": "hash-a"}
    validate_submission_records([valid], ["a"])

    with pytest.raises(SubmissionValidationError, match="coverage"):
        validate_submission_records([valid], ["a", "b"])
    with pytest.raises(SubmissionValidationError, match="duplicate"):
        validate_submission_records([valid, valid], ["a", "a"])
    with pytest.raises(SubmissionValidationError, match="malformed"):
        validate_submission_records(
            [{"id": "a", "tool_calls": [{"name": "x"}], "state_hash": "hash"}], ["a"]
        )


def test_parallel_batch_writes_original_order_even_when_first_sample_finishes_last(tmp_path: Path) -> None:
    dataset = _batch_dataset(tmp_path)

    class SlowSource(ExplicitActionSource):
        def next_action(self, context):
            if context.sample.id == "public_0000":
                time.sleep(0.02)
            return super().next_action(context)

    output = tmp_path / "parallel" / "submission.jsonl"
    BatchRunner(
        dataset,
        lambda _sample: SlowSource([]),
        BatchConfig(output=output, input_mode="text", workers=3, validate_official=False),
        runtime=Runtime(dataset, session_factory=PerSampleFactory()),
        transcript_provider=lambda sample: sample.id,
        progress_reporter=lambda *_args: None,
    ).run()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["public_0000", "public_0001", "public_0002"]


def test_terminal_agent_failure_is_reported_without_submission_row(tmp_path: Path) -> None:
    dataset = _batch_dataset(tmp_path, count=1)
    output = tmp_path / "failed-agent" / "submission.jsonl"

    def decisions(_sample):
        return LlmDecisionSource(
            ScriptedModelClient([ModelResponse(content="not-json")]),
            InferenceSettings(model="fixture"),
            max_invalid_proposals=0,
        )

    summary = BatchRunner(
        dataset,
        decisions,
        BatchConfig(output=output, input_mode="text", validate_official=False),
        runtime=Runtime(dataset, session_factory=PerSampleFactory()),
        transcript_provider=lambda sample: sample.id,
        progress_reporter=lambda *_args: None,
    ).run()

    assert not summary.ready_for_submission
    assert not output.exists()
    failure = json.loads(summary.failures.read_text(encoding="utf-8"))
    assert failure["termination_reason"] == "model_bad_output_limit"
    metrics = json.loads(summary.metrics.read_text(encoding="utf-8"))
    assert metrics["invalid_model_proposals"] == 1
    assert metrics["termination_reasons"] == {"model_bad_output_limit": 1}
