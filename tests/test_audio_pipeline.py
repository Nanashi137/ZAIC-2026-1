from __future__ import annotations

import json
from pathlib import Path

from lifemate.agent import LlmDecisionSource
from lifemate.asr import Qwen3ASRSettings, TranscriptionResult
from lifemate.audio_pipeline import AudioInferencePipeline
from lifemate.model import InferenceSettings, ModelResponse, ScriptedModelClient
from lifemate.runtime import ExplicitActionSource, Runtime, ToolAction
from tests.support import FakeFactory, FakeSession, make_dataset


class FakeTranscriber:
    settings = Qwen3ASRSettings(checkpoint="fixture/Qwen3-ASR")

    def __init__(self, transcripts: dict[str, str]) -> None:
        self.transcripts = transcripts
        self.calls: list[str] = []

    def transcribe(self, audio_path: Path | str, *, deadline=None) -> TranscriptionResult:
        del deadline
        name = Path(audio_path).name
        self.calls.append(name)
        return TranscriptionResult(
            text=self.transcripts[name],
            language="Vietnamese",
            inference_latency_seconds=0.25,
            audio_sha256=name,
            checkpoint=self.settings.checkpoint,
            backend=self.settings.backend,
        )


def _audio_files(dataset, names: list[str]) -> None:
    directory = dataset.split_dir / "audio" / "public_test_1"
    directory.mkdir(parents=True)
    for name in names:
        (directory / name).write_bytes(b"fixture")


def test_final_and_history_audio_order_and_assistant_records_are_preserved(tmp_path: Path) -> None:
    history = [
        {"role": "user", "audio": "audio/public_test_1/earlier_1.wav"},
        {
            "role": "assistant",
            "text": "Đã kiểm tra.",
            "tool_calls": [{"name": "alarm.list", "arguments": {}}],
            "tool_results": [{"ok": True, "alarms": []}],
        },
        {"role": "user", "audio": "audio/public_test_1/earlier_2.wav"},
    ]
    dataset = make_dataset(tmp_path, history=history)
    _audio_files(dataset, ["public_test_1.wav", "earlier_1.wav", "earlier_2.wav"])
    session = FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"})
    runtime = Runtime(dataset, session_factory=FakeFactory(session))
    transcriber = FakeTranscriber(
        {
            "public_test_1.wav": "Yêu cầu cuối",
            "earlier_1.wav": "Yêu cầu trước một",
            "earlier_2.wav": "Yêu cầu trước hai",
        }
    )

    result = AudioInferencePipeline(dataset, runtime, transcriber).run(
        "public_test_1", ExplicitActionSource([ToolAction("alarm.list", {})])
    )

    assert transcriber.calls == ["public_test_1.wav", "earlier_1.wav", "earlier_2.wav"]
    assert result.run.transcript == "Yêu cầu cuối"
    assert [turn["role"] for turn in result.conversation_history] == ["user", "assistant", "user"]
    assert result.conversation_history[0]["transcript"] == "Yêu cầu trước một"
    assert result.conversation_history[1]["text"] == "Đã kiểm tra."
    assert result.conversation_history[1]["tool_calls"][0]["name"] == "alarm.list"
    assert result.conversation_history[1]["tool_results"][0]["alarms"] == ()
    assert result.diagnostics.transcribed_turns == 3
    assert result.diagnostics.final_audio_latency_seconds == 0.25
    assert result.diagnostics.history_audio_latency_seconds == 0.5


def test_history_is_model_context_only_and_historical_tools_are_not_replayed(tmp_path: Path) -> None:
    history = [
        {"role": "user", "audio": "audio/public_test_1/earlier.wav"},
        {
            "role": "assistant",
            "text": "Đã tạo.",
            "tool_calls": [{"name": "alarm.create", "arguments": {"time": "07:00"}}],
            "tool_results": [{"ok": True}],
        },
    ]
    dataset = make_dataset(tmp_path, history=history)
    _audio_files(dataset, ["public_test_1.wav", "earlier.wav"])
    session = FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"})
    client = ScriptedModelClient([ModelResponse(content='{"status":"done","reason":"complete"}')])
    source = LlmDecisionSource(client, InferenceSettings(model="fixture"))
    transcriber = FakeTranscriber({"public_test_1.wav": "Xong chưa?", "earlier.wav": "Đặt lúc bảy giờ"})

    result = AudioInferencePipeline(
        dataset,
        Runtime(dataset, session_factory=FakeFactory(session)),
        transcriber,
    ).run("public_test_1", source)

    assert session.calls == []
    assert result.run.submission["tool_calls"] == []
    payload = json.loads(client.requests[0].messages[1]["content"])
    assert payload["final_user_transcript"] == "Xong chưa?"
    assert payload["final_user_transcript_source"] == "qwen3_asr"
    assert payload["history"][0]["transcript"] == "Đặt lúc bảy giờ"
    assert payload["history"][1]["tool_calls_already_applied"][0]["name"] == "alarm.create"
    assert "earlier.wav" not in client.requests[0].messages[1]["content"]


def test_text_only_phase02_path_remains_functional(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    session = FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"})

    result = Runtime(dataset, session_factory=FakeFactory(session)).run(
        "public_test_1",
        "debug transcript",
        ExplicitActionSource([ToolAction("alarm.list", {})]),
    )

    assert result.transcript == "debug transcript"
    assert result.submission["tool_calls"] == [{"name": "alarm.list", "arguments": {}}]


def test_audio_pipeline_without_history_transcribes_only_final_turn(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    _audio_files(dataset, ["public_test_1.wav"])
    session = FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"})
    transcriber = FakeTranscriber({"public_test_1.wav": "Liệt kê báo thức"})

    result = AudioInferencePipeline(
        dataset,
        Runtime(dataset, session_factory=FakeFactory(session)),
        transcriber,
    ).run("public_test_1", ExplicitActionSource([]))

    assert transcriber.calls == ["public_test_1.wav"]
    assert result.conversation_history == ()
    assert result.diagnostics.transcribed_turns == 1
