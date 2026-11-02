from __future__ import annotations

from pathlib import Path

import pytest

from lifemate.agent import LlmDecisionSource, ModelToolRegistry
from lifemate.asr import Qwen3ASRSettings, TranscriptionResult
from lifemate.audio_pipeline import AudioInferencePipeline
from lifemate.data import DatasetReader
from lifemate.model import InferenceSettings, ModelResponse, ModelToolCall, ScriptedModelClient
from lifemate.runtime import ExplicitActionSource, Runtime, ToolAction
from lifemate.sdk import OfficialSessionFactory
from lifemate.tool_schema import FORBIDDEN_ROOT_KEYWORDS

DATASET_ROOT = Path("data/train+public_test")
SAMPLE_ID = "public_0039"


pytestmark = pytest.mark.integration


class _PublicAudioFakeTranscriber:
    settings = Qwen3ASRSettings(checkpoint="fixture/Qwen3-ASR")

    def __init__(self) -> None:
        self.paths: list[Path] = []

    def transcribe(self, audio_path: Path | str, *, deadline=None) -> TranscriptionResult:
        del deadline
        path = Path(audio_path)
        assert path.is_file()
        self.paths.append(path)
        return TranscriptionResult(
            text=f"fixture transcript for {path.stem}",
            language="Vietnamese",
            inference_latency_seconds=0.0,
            audio_sha256=path.name,
            checkpoint=self.settings.checkpoint,
            backend=self.settings.backend,
        )


def test_real_sdk_read_produces_official_submission_shape() -> None:
    result = Runtime(DatasetReader(DATASET_ROOT)).run(
        SAMPLE_ID,
        "liệt kê báo thức (debug override)",
        ExplicitActionSource([ToolAction("alarm.list", {})]),
    )

    assert result.steps[0].status == "read_succeeded"
    assert result.submission["id"] == SAMPLE_ID
    assert result.submission["tool_calls"] == [{"name": "alarm.list", "arguments": {}}]
    assert len(result.submission["state_hash"]) == 64


def test_real_sdk_records_rejected_call_and_enforces_x_one_of() -> None:
    dataset = DatasetReader(DATASET_ROOT)
    factory = OfficialSessionFactory()

    # Independent test session: this intentionally bypasses runtime preflight to verify SDK recording.
    rejected = factory.create("public_test", SAMPLE_ID, dataset.dataset_root)
    observation = rejected.call("alarm.create", {})
    assert observation["ok"] is False
    assert rejected.get_tool_calls() == [{"name": "alarm.create", "arguments": {}}]

    # Independent test session for validator behavior; no call is executed.
    phone_sample = dataset.get_sample("public_0003")
    phone = factory.create("public_test", phone_sample.id, dataset.dataset_root)
    assert phone.validate("phone.call", {})["ok"] is False
    assert phone.validate("phone.call", {"phone_number": "0900000000"})["ok"] is True
    assert phone.get_tool_calls() == []


def test_all_catalog_tools_have_openai_compatible_roots_and_contacts_update_keeps_sdk_validation() -> None:
    dataset = DatasetReader(DATASET_ROOT)
    catalog = dataset.load_catalog()
    registry = ModelToolRegistry.from_allowed(catalog.tools)

    assert len(catalog.tools) == 122
    assert len(registry.definitions) == 123
    for definition in registry.definitions:
        parameters = definition["function"]["parameters"]
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        assert not FORBIDDEN_ROOT_KEYWORDS.intersection(parameters)

    model_name = registry.official_to_model["contacts.update"]
    contacts_parameters = next(
        definition["function"]["parameters"]
        for definition in registry.definitions
        if definition["function"]["name"] == model_name
    )
    assert set(contacts_parameters["properties"]) == {"contact_id", "phone", "email"}
    assert contacts_parameters["required"] == ["contact_id"]
    assert "phone OR email" in contacts_parameters["description"]

    sample = next(sample for sample in dataset.iter_samples() if "contacts.update" in sample.allowed_tool_names)
    session = OfficialSessionFactory().create("public_test", sample.id, dataset.dataset_root)
    assert session.validate("contacts.update", {"contact_id": "fixture"})["ok"] is False
    assert session.validate(
        "contacts.update", {"contact_id": "fixture", "phone": "0900000000"}
    )["ok"] is True
    assert session.get_tool_calls() == []


def test_real_sdk_system_tool_and_noop_contract() -> None:
    dataset = DatasetReader(DATASET_ROOT)
    factory = OfficialSessionFactory()

    meta = factory.create("public_test", SAMPLE_ID, dataset.dataset_root)
    assert "sys.ask_missing_info" in meta.allowed
    assert meta.validate("alarm.create", {"time": 700})["ok"] is False
    assert meta.validate(
        "sys.ask_missing_info", {"tool_name": "alarm.create", "params": ["time"]}
    )["ok"] is True
    assert meta.call(
        "sys.ask_missing_info", {"tool_name": "alarm.create", "params": ["time"]}
    ) == {"ok": True, "waiting_for_user": True}

    noop = factory.create("public_test", SAMPLE_ID, dataset.dataset_root)
    before = noop.get_state_hash()
    assert noop.validate("alarm.update", {"alarm_id": "definitely_missing"})["ok"] is True
    observation = noop.call("alarm.update", {"alarm_id": "definitely_missing"})
    assert observation.get("ok") is True and observation.get("done") is False
    assert noop.get_state_hash() == before


def test_real_sdk_scripted_model_loop_uses_dynamic_tools_and_official_outputs() -> None:
    client = ScriptedModelClient(
        [
            ModelResponse(tool_calls=(ModelToolCall("read_alarm", "alarm__list", "{}"),)),
            ModelResponse(
                tool_calls=(
                    ModelToolCall(
                        "finish", "agent__finish", '{"status":"done","reason":"alarms listed"}'
                    ),
                )
            ),
        ]
    )
    source = LlmDecisionSource(client, InferenceSettings(model="offline-scripted"))

    result = Runtime(DatasetReader(DATASET_ROOT)).run(
        SAMPLE_ID,
        "liệt kê báo thức (debug override)",
        source,
    )

    assert result.termination_reason == "model_completed"
    assert result.submission["tool_calls"] == [{"name": "alarm.list", "arguments": {}}]
    assert len(result.submission["state_hash"]) == 64
    first_tools = {tool["function"]["name"] for tool in client.requests[0].tools}
    assert "alarm__list" in first_tools
    assert "sys__ask_missing_info" in first_tools
    assert "agent__finish" in first_tools


def test_audio_history_pipeline_uses_real_toolenv_without_replaying_history() -> None:
    dataset = DatasetReader(DATASET_ROOT)
    transcriber = _PublicAudioFakeTranscriber()
    result = AudioInferencePipeline(dataset, Runtime(dataset), transcriber).run(
        "public_0006", ExplicitActionSource([])
    )

    assert [path.name for path in transcriber.paths] == [
        "public_0006.wav",
        "public_0006_u1.wav",
        "public_0006_u2.wav",
    ]
    assert result.run.submission["tool_calls"] == []
    assert len(result.run.submission["state_hash"]) == 64
    assert result.diagnostics.transcribed_turns == 3
