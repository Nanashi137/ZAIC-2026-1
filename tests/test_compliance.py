from __future__ import annotations

import ast
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from lifemate.agent import LlmDecisionSource
from lifemate.asr import Qwen3ASRSettings
from lifemate.batch import BatchConfig, BatchRunner
from lifemate.cli import build_batch_parser, build_parser, main
from lifemate.compliance import (
    SampleDeadline,
    SampleDeadlineExceeded,
    configure_greedy_generation,
    configure_reproducibility,
)
from lifemate.model import (
    InferenceSettings,
    ModelClientError,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    OpenAICompatibleClient,
    RetrySettings,
    ScriptedModelClient,
)
from lifemate.runtime import ExplicitActionSource, Runtime, ToolAction
from tests.support import FakeFactory, FakeSession, make_dataset


class FakeTime:
    def __init__(self) -> None:
        self.value = 0.0

    def clock(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


class NeverValidator:
    def __init__(self) -> None:
        self.calls = 0

    def validate(self, _submission_path: Path) -> str:
        self.calls += 1
        raise AssertionError("competition mode must not run the post-batch validator")


def test_mocked_llm_over_deadline_is_a_failed_sample_with_one_toolenv(tmp_path: Path) -> None:
    fake_time = FakeTime()
    dataset = make_dataset(tmp_path)
    session = FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"})
    factory = FakeFactory(session)

    class SlowSource(ExplicitActionSource):
        def __init__(self) -> None:
            super().__init__([])
            self.deadline = None

        def set_deadline(self, deadline) -> None:
            self.deadline = deadline

        def next_action(self, context):
            del context
            fake_time.advance(3.0)
            self.deadline.check("mocked LLM")

    output = tmp_path / "timeout" / "submission.jsonl"
    summary = BatchRunner(
        dataset,
        lambda _sample: SlowSource(),
        BatchConfig(
            output=output,
            input_mode="text",
            competition_mode=True,
            sample_timeout_seconds=2.0,
        ),
        runtime=Runtime(dataset, session_factory=factory),
        transcript_provider=lambda _sample: "debug",
        official_validator=NeverValidator(),
        progress_reporter=lambda *_args: None,
        clock=fake_time.clock,
        deadline_clock=fake_time.clock,
    ).run()

    assert summary.failed_samples == 1
    assert not summary.ready_for_submission
    assert not output.exists()
    assert len(factory.creations) == 1
    failure = json.loads(summary.failures.read_text(encoding="utf-8"))
    assert failure["termination_reason"] == "sample_timeout"
    assert json.loads((output.parent / "metrics.json").read_text(encoding="utf-8"))[
        "termination_reasons"
    ] == {"sample_timeout": 1}


def test_mocked_asr_over_deadline_fails_before_toolenv_creation(tmp_path: Path) -> None:
    fake_time = FakeTime()
    dataset = make_dataset(tmp_path)
    audio = dataset.split_dir / "audio/public_test_1/public_test_1.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"fixture")
    factory = FakeFactory(FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"}))

    class SlowTranscriber:
        settings = Qwen3ASRSettings(checkpoint="fixture/Qwen3-ASR")

        def transcribe(self, _audio_path, *, deadline=None):
            fake_time.advance(3.0)
            deadline.check("mocked ASR")

    output = tmp_path / "asr-timeout" / "submission.jsonl"
    summary = BatchRunner(
        dataset,
        lambda _sample: ExplicitActionSource([]),
        BatchConfig(
            output=output,
            input_mode="audio",
            competition_mode=True,
            sample_timeout_seconds=2.0,
        ),
        runtime=Runtime(dataset, session_factory=factory),
        transcriber=SlowTranscriber(),
        official_validator=NeverValidator(),
        progress_reporter=lambda *_args: None,
        clock=fake_time.clock,
        deadline_clock=fake_time.clock,
    ).run()

    assert summary.failed_samples == 1
    assert factory.creations == []
    assert not output.exists()
    assert (output.parent / "batch-progress.jsonl").read_text(encoding="utf-8") == ""


def test_multiple_agent_steps_receive_one_diminishing_deadline(tmp_path: Path) -> None:
    fake_time = FakeTime()
    remaining: list[float] = []

    class BudgetClient:
        name = "budget-client"

        def __init__(self) -> None:
            self.index = 0

        def complete(self, request: ModelRequest) -> ModelResponse:
            assert request.deadline is not None
            remaining.append(request.deadline.remaining_seconds())
            fake_time.advance(1.0)
            self.index += 1
            if self.index == 1:
                return ModelResponse(tool_calls=(ModelToolCall("read", "alarm__list", "{}"),))
            return ModelResponse(
                tool_calls=(
                    ModelToolCall(
                        "finish", "agent__finish", '{"status":"done","reason":"complete"}'
                    ),
                )
            )

    dataset = make_dataset(tmp_path)
    factory = FakeFactory(FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"}))
    deadline = SampleDeadline.start(5.0, clock=fake_time.clock)
    source = LlmDecisionSource(BudgetClient(), InferenceSettings(model="fixture"), deadline=deadline)

    result = Runtime(dataset, session_factory=factory).run(
        "public_test_1",
        "list alarms",
        source,
        deadline=deadline,
    )

    assert result.termination_reason == "model_completed"
    assert remaining == [5.0, 4.0]
    assert len(factory.creations) == 1


def test_model_timeout_is_capped_and_retry_stops_when_budget_is_insufficient() -> None:
    fake_time = FakeTime()

    class FlakyTransport:
        def __init__(self) -> None:
            self.timeouts: list[float] = []

        def post(self, _url, _payload, _headers, timeout_seconds):
            self.timeouts.append(timeout_seconds)
            raise ModelClientError("transient", transient=True)

    transport = FlakyTransport()
    client = OpenAICompatibleClient(
        base_url="http://localhost:8000/v1",
        timeout_seconds=60.0,
        retry_settings=RetrySettings(max_retries=3, base_delay_seconds=1.0, max_delay_seconds=8.0),
        transport=transport,
        sleep=fake_time.sleep,
        clock=fake_time.clock,
    )
    deadline = SampleDeadline.start(1.0, clock=fake_time.clock)

    with pytest.raises(SampleDeadlineExceeded, match="insufficient time"):
        client.complete(
            ModelRequest(
                messages=(),
                tools=(),
                settings=InferenceSettings(model="fixture"),
                deadline=deadline,
            )
        )
    assert transport.timeouts == [1.0]


def test_insufficient_remaining_time_prevents_new_model_request(tmp_path: Path) -> None:
    fake_time = FakeTime()
    client = ScriptedModelClient(
        [ModelResponse(tool_calls=(ModelToolCall("read", "alarm__list", "{}"),))]
    )
    deadline = SampleDeadline.start(0.2, clock=fake_time.clock)
    source = LlmDecisionSource(
        client,
        InferenceSettings(model="fixture"),
        deadline=deadline,
        minimum_model_time_seconds=0.25,
    )
    dataset = make_dataset(tmp_path)

    result = Runtime(
        dataset,
        session_factory=FakeFactory(
            FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"})
        ),
    ).run("public_test_1", "list alarms", source, deadline=deadline)

    assert result.termination_reason == "sample_timeout"
    assert client.requests == []


def test_deadline_after_successful_write_preserves_official_trace(tmp_path: Path) -> None:
    fake_time = FakeTime()
    dataset = make_dataset(tmp_path)

    def slow_success(_name, _arguments):
        fake_time.advance(3.0)
        return {"ok": True}

    session = FakeSession(
        {"alarm.list", "alarm.create", "sys.ask_missing_info"}, call=slow_success
    )
    deadline = SampleDeadline.start(2.0, clock=fake_time.clock)
    result = Runtime(dataset, session_factory=FakeFactory(session)).run(
        "public_test_1",
        "create alarm",
        ExplicitActionSource([ToolAction("alarm.create", {"time": "07:00"})]),
        deadline=deadline,
    )

    assert result.termination_reason == "sample_timeout"
    assert result.submission["tool_calls"] == [
        {"name": "alarm.create", "arguments": {"time": "07:00"}}
    ]
    assert result.submission["state_hash"] == "official-hash"


def test_fixed_seed_repeats_python_numpy_and_torch_configuration(monkeypatch) -> None:
    numpy_seeds: list[int] = []
    torch_seeds: list[int] = []
    cuda_seeds: list[int] = []
    deterministic_calls: list[tuple[bool, bool]] = []
    fake_numpy = SimpleNamespace(random=SimpleNamespace(seed=numpy_seeds.append))
    fake_cudnn = SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True)
    fake_matmul = SimpleNamespace(allow_tf32=True)
    fake_torch = SimpleNamespace(
        manual_seed=torch_seeds.append,
        cuda=SimpleNamespace(is_available=lambda: True, manual_seed_all=cuda_seeds.append),
        use_deterministic_algorithms=lambda enabled, warn_only: deterministic_calls.append((enabled, warn_only)),
        backends=SimpleNamespace(
            cudnn=fake_cudnn,
            cuda=SimpleNamespace(matmul=fake_matmul),
        ),
    )
    monkeypatch.setitem(sys.modules, "numpy", fake_numpy)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    first = configure_reproducibility(42)
    sequence_one = [random.random() for _ in range(3)]
    second = configure_reproducibility(42)
    sequence_two = [random.random() for _ in range(3)]

    assert sequence_one == sequence_two
    assert numpy_seeds == [42, 42]
    assert torch_seeds == [42, 42]
    assert cuda_seeds == [42, 42]
    assert deterministic_calls == [(True, True), (True, True)]
    assert first == second
    assert fake_cudnn.benchmark is False and fake_cudnn.deterministic is True
    assert fake_cudnn.allow_tf32 is False and fake_matmul.allow_tf32 is False


def test_greedy_generation_configuration_is_explicit() -> None:
    config = SimpleNamespace(do_sample=True, num_beams=4)
    assert configure_greedy_generation(SimpleNamespace(generation_config=config))
    assert config.do_sample is False
    assert config.num_beams == 1
    assert not configure_greedy_generation(SimpleNamespace())


def test_competition_mode_uses_one_toolenv_and_skips_post_batch_validator(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    factory = FakeFactory(FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"}))
    validator = NeverValidator()
    output = tmp_path / "competition" / "submission.jsonl"

    summary = BatchRunner(
        dataset,
        lambda _sample: ExplicitActionSource([]),
        BatchConfig(output=output, input_mode="text", competition_mode=True),
        runtime=Runtime(dataset, session_factory=factory),
        transcript_provider=lambda _sample: "debug",
        official_validator=validator,
        progress_reporter=lambda *_args: None,
    ).run()

    assert summary.ready_for_submission
    assert len(factory.creations) == 1
    assert validator.calls == 0
    assert json.loads(output.read_text(encoding="utf-8"))["state_hash"] == "official-hash"
    assert json.loads(summary.metrics.read_text(encoding="utf-8"))["official_schema_validation"] is None


def test_inference_sources_do_not_access_prohibited_environment_paths() -> None:
    source_paths = [*Path("lifemate").glob("*.py"), Path("scripts/inspect_kit.py")]
    forbidden_literals: list[tuple[Path, str]] = []
    forbidden_calls: list[tuple[Path, str]] = []
    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lowered = node.value.lower()
                if "_environment/" in lowered or "_environment\\" in lowered or "state.json" in lowered:
                    forbidden_literals.append((path, node.value))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get_state":
                forbidden_calls.append((path, "get_state"))
    assert forbidden_literals == []
    assert forbidden_calls == []


def test_cli_compatibility_and_competition_defaults() -> None:
    batch = build_batch_parser().parse_args(["--output", "outputs/submission.jsonl"])
    assert batch.competition_mode is False
    assert batch.sample_timeout_seconds == 28.0
    assert batch.seed == 42

    competition = build_batch_parser().parse_args(
        ["--output", "outputs/submission.jsonl", "--competition-mode"]
    )
    config = BatchConfig(
        output=competition.output,
        competition_mode=competition.competition_mode,
        sample_timeout_seconds=competition.sample_timeout_seconds,
    )
    assert config.sample_timeout_seconds == 28.0
    assert config.validate_official is False

    single = build_parser().parse_args(["--sample-id", "public_0001", "--transcript", "debug"])
    assert single.seed == 42
    assert single.sample_timeout_seconds == 28.0

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--sample-id",
                "public_0001",
                "--transcript",
                "debug",
                "--mode",
                "model",
                "--competition-mode",
                "--base-url",
                "https://api.openai.com/v1",
                "--allow-external-data",
            ]
        )
