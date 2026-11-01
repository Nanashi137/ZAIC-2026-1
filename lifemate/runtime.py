"""Sequential, bounded tool runtime shared by deterministic and LLM decision sources."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from .compliance import SampleDeadline, SampleDeadlineExceeded
from .data import DatasetReader, Sample
from .sdk import OfficialSessionFactory, SessionFactory, ToolSession

_EXCEPTION_MESSAGE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception):")


class ContractMismatch(RuntimeError):
    """Public input permissions disagree with the official environment metadata."""


class DecisionError(RuntimeError):
    """Base class for a safe decision-source failure."""


class BadDecisionOutput(DecisionError):
    """A malformed model proposal that may be retried without executing a tool."""


class DecisionRetryLimit(DecisionError):
    """The model exceeded the configured invalid-proposal limit."""


class DecisionRefusal(DecisionError):
    """The model explicitly refused to make a decision."""


class DecisionModelError(DecisionError):
    """The model provider failed before a tool decision was available."""


@dataclass(frozen=True)
class ToolAction:
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    provider_call_id: str | None = None
    model_name: str | None = None

    def as_arguments(self) -> dict[str, Any]:
        return dict(self.arguments)


@dataclass(frozen=True)
class StepResult:
    index: int
    action: ToolAction | None
    status: str
    validation: Mapping[str, Any] | None
    observation: Mapping[str, Any] | None
    error: str | None = None


@dataclass(frozen=True)
class DecisionContext:
    transcript: str
    sample: Sample
    allowed_tools: Mapping[str, Mapping[str, Any]]
    steps: tuple[StepResult, ...]
    conversation_history: tuple[Mapping[str, Any], ...] | None = None


class DecisionSource(Protocol):
    name: str

    def next_action(self, context: DecisionContext) -> ToolAction | None: ...


class ExplicitActionSource:
    """Deterministic Phase 01 stub: return caller-provided actions in order."""

    name = "explicit_actions"

    def __init__(self, actions: list[ToolAction] | tuple[ToolAction, ...]) -> None:
        self._actions = tuple(actions)
        self._index = 0

    def next_action(self, context: DecisionContext) -> ToolAction | None:
        del context
        if self._index >= len(self._actions):
            return None
        action = self._actions[self._index]
        self._index += 1
        return action


@dataclass(frozen=True)
class RuntimeConfig:
    max_steps: int = 8
    max_tool_calls: int = 6

    def __post_init__(self) -> None:
        if self.max_steps < 1 or self.max_tool_calls < 1:
            raise ValueError("max_steps and max_tool_calls must be positive")


@dataclass(frozen=True)
class RunResult:
    sample_id: str
    transcript: str
    decision_source: str
    termination_reason: str
    steps: tuple[StepResult, ...]
    tool_calls_sent: int
    model_usage: Mapping[str, int] | None
    model_latency_seconds: float | None
    submission: Mapping[str, Any]


class Runtime:
    def __init__(
        self,
        dataset: DatasetReader,
        *,
        session_factory: SessionFactory | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.session_factory = session_factory or OfficialSessionFactory()
        self.config = config or RuntimeConfig()

    def run(
        self,
        sample_id: str,
        transcript: str,
        decisions: DecisionSource,
        *,
        conversation_history: tuple[Mapping[str, Any], ...] | None = None,
        deadline: SampleDeadline | None = None,
    ) -> RunResult:
        if deadline is not None:
            deadline.check("sample setup")
        if not transcript.strip():
            raise ValueError("an explicit non-empty debug transcript is required")
        sample = self.dataset.get_sample(sample_id)
        catalog = self.dataset.load_catalog()
        allowed_tools = catalog.restricted_to(sample.allowed_tool_names)
        if deadline is not None:
            deadline.check("ToolEnv creation")

        # This is the sole ToolEnv construction for this inference.
        session = self.session_factory.create(self.dataset.split, sample.id, self.dataset.dataset_root)
        self._verify_permissions(sample, session)

        steps: list[StepResult] = []
        calls_sent = 0
        executed_writes: set[str] = set()
        termination = "max_steps_exhausted"
        for index in range(self.config.max_steps):
            try:
                if deadline is not None:
                    deadline.check("agent step")
            except SampleDeadlineExceeded as exc:
                steps.append(StepResult(index, None, "sample_timeout", None, None, str(exc)))
                termination = "sample_timeout"
                break
            context = DecisionContext(
                transcript=transcript,
                sample=sample,
                allowed_tools=allowed_tools,
                steps=tuple(steps),
                conversation_history=conversation_history,
            )
            try:
                action = decisions.next_action(context)
            except BadDecisionOutput as exc:
                steps.append(StepResult(index, None, "model_bad_output", None, None, str(exc)))
                continue
            except DecisionRetryLimit as exc:
                steps.append(StepResult(index, None, "model_bad_output_limit", None, None, str(exc)))
                termination = "model_bad_output_limit"
                break
            except DecisionRefusal as exc:
                steps.append(StepResult(index, None, "model_refusal", None, None, str(exc)))
                termination = "model_refusal"
                break
            except DecisionModelError as exc:
                steps.append(StepResult(index, None, "model_error", None, None, str(exc)))
                termination = "model_error"
                break
            except SampleDeadlineExceeded as exc:
                steps.append(StepResult(index, None, "sample_timeout", None, None, str(exc)))
                termination = "sample_timeout"
                break
            if action is None:
                try:
                    if deadline is not None:
                        deadline.check("agent completion")
                except SampleDeadlineExceeded as exc:
                    steps.append(StepResult(index, None, "sample_timeout", None, None, str(exc)))
                    termination = "sample_timeout"
                    break
                termination = str(getattr(decisions, "finish_reason", "decision_source_done"))
                break
            try:
                if deadline is not None:
                    deadline.check("tool validation")
            except SampleDeadlineExceeded as exc:
                steps.append(StepResult(index, action, "sample_timeout", None, None, str(exc)))
                termination = "sample_timeout"
                break
            if action.name not in allowed_tools:
                steps.append(StepResult(index, action, "permission_rejected", None, None))
                continue

            arguments = action.as_arguments()
            validation = session.validate(action.name, arguments)
            if not validation.get("ok"):
                steps.append(
                    StepResult(index, action, "schema_rejected", MappingProxyType(dict(validation)), None)
                )
                continue
            if calls_sent >= self.config.max_tool_calls:
                termination = "tool_call_budget_exhausted"
                break

            tool_kind = allowed_tools[action.name].get("kind")
            write_key = _action_key(action.name, arguments) if tool_kind == "write" else None
            if write_key is not None and write_key in executed_writes:
                steps.append(
                    StepResult(
                        index,
                        action,
                        "duplicate_write_blocked",
                        MappingProxyType(dict(validation)),
                        None,
                        "an identical write already reached the SDK in this inference",
                    )
                )
                termination = "duplicate_write_blocked"
                break
            if write_key is not None:
                executed_writes.add(write_key)
            try:
                if deadline is not None:
                    deadline.check("tool execution")
            except SampleDeadlineExceeded as exc:
                steps.append(
                    StepResult(index, action, "sample_timeout", MappingProxyType(dict(validation)), None, str(exc))
                )
                termination = "sample_timeout"
                break
            calls_sent += 1
            try:
                observation = session.call(action.name, arguments)
            except Exception as exc:
                steps.append(
                    StepResult(
                        index,
                        action,
                        "execution_uncertain",
                        MappingProxyType(dict(validation)),
                        None,
                        f"{type(exc).__name__}: {exc}",
                    )
                )
                termination = "execution_uncertain"
                break
            status = _classify_observation(action.name, allowed_tools[action.name], observation)
            steps.append(
                StepResult(
                    index,
                    action,
                    status,
                    MappingProxyType(dict(validation)),
                    MappingProxyType(dict(observation)),
                )
            )
            try:
                if deadline is not None:
                    deadline.check("tool execution")
            except SampleDeadlineExceeded:
                termination = "sample_timeout"
                break
            if observation.get("waiting_for_user") is True:
                termination = "awaiting_user"
                break
            if status in {"sdk_rejected", "tool_error"} and tool_kind == "write":
                termination = status
                break
            if calls_sent >= self.config.max_tool_calls:
                termination = "tool_call_budget_exhausted"
                break

        submission = MappingProxyType(
            {
                "id": sample.id,
                "tool_calls": session.get_tool_calls(),
                "state_hash": session.get_state_hash(),
            }
        )
        try:
            if deadline is not None:
                deadline.check("sample output finalization")
        except SampleDeadlineExceeded as exc:
            steps.append(StepResult(len(steps), None, "sample_timeout", None, None, str(exc)))
            termination = "sample_timeout"
        return RunResult(
            sample_id=sample.id,
            transcript=transcript,
            decision_source=decisions.name,
            termination_reason=termination,
            steps=tuple(steps),
            tool_calls_sent=calls_sent,
            model_usage=_model_usage(decisions),
            model_latency_seconds=_model_latency(decisions),
            submission=submission,
        )

    @staticmethod
    def _verify_permissions(sample: Sample, session: ToolSession) -> None:
        official = frozenset(session.allowed)
        expected = sample.allowed_tool_names
        if official != expected:
            missing = sorted(expected - official)
            extra = sorted(official - expected)
            raise ContractMismatch(f"input/SDK allowed-tool mismatch; missing={missing}, extra={extra}")


def _classify_observation(name: str, tool_spec: Mapping[str, Any], observation: Mapping[str, Any]) -> str:
    if observation.get("ok") is False:
        return "sdk_rejected"
    if observation.get("done") is False:
        message = observation.get("message")
        if isinstance(message, str) and _EXCEPTION_MESSAGE.match(message):
            return "tool_error"
        return "no_op"
    kind = tool_spec.get("kind")
    if kind == "write":
        return "write_succeeded"
    if kind == "read":
        return "read_succeeded"
    if name == "sys.ask_missing_info" or kind == "meta":
        return "meta_succeeded"
    return "executed"


def _action_key(name: str, arguments: Mapping[str, Any]) -> str:
    return name + "\0" + json.dumps(arguments, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)


def _model_usage(decisions: DecisionSource) -> Mapping[str, int] | None:
    usage = getattr(decisions, "usage", None)
    if isinstance(usage, Mapping) and all(
        isinstance(key, str) and isinstance(value, int) and not isinstance(value, bool)
        for key, value in usage.items()
    ):
        return MappingProxyType(dict(usage))
    return None


def _model_latency(decisions: DecisionSource) -> float | None:
    latency = getattr(decisions, "latency_seconds", None)
    if isinstance(latency, (int, float)) and not isinstance(latency, bool) and latency >= 0:
        return float(latency)
    return None
