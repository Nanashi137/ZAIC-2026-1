"""Sequential, bounded tool runtime shared by deterministic and LLM decision sources."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from .compliance import SampleDeadline, SampleDeadlineExceeded
from .data import DatasetReader, Sample
from .sdk import OfficialSessionFactory, SessionFactory, ToolSession

_EXCEPTION_MESSAGE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception):")
_NUMERIC_REPEAT = re.compile(r"\b(\d+)\s*(?:times?|l[aầ]n)\b", re.IGNORECASE)
_WORD_REPEATS = {
    "twice": 2,
    "two times": 2,
    "three times": 3,
    "four times": 4,
    "hai lần": 2,
    "ba lần": 3,
    "bốn lần": 4,
}
_REPEAT_ONCE_MORE = re.compile(r"\b(?:repeat(?:\s+it)?|l[aặ]p\s+l[aạ]i)\b", re.IGNORECASE)
AGENT_FINISH = "agent.finish"


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
    model_request_diagnostics: tuple[Mapping[str, Any], ...]
    phase_timings: Mapping[str, float]


class Runtime:
    def __init__(
        self,
        dataset: DatasetReader,
        *,
        session_factory: SessionFactory | None = None,
        config: RuntimeConfig | None = None,
        clock: Any = time.monotonic,
    ) -> None:
        self.dataset = dataset
        self.session_factory = session_factory or OfficialSessionFactory()
        self.config = config or RuntimeConfig()
        self.clock = clock

    def run(
        self,
        sample_id: str,
        transcript: str,
        decisions: DecisionSource,
        *,
        conversation_history: tuple[Mapping[str, Any], ...] | None = None,
        deadline: SampleDeadline | None = None,
        session_observer: Callable[[ToolSession], None] | None = None,
    ) -> RunResult:
        run_started = self.clock()
        phase_timings = {
            "sample_setup_seconds": 0.0,
            "toolenv_creation_seconds": 0.0,
            "decision_seconds": 0.0,
            "validation_seconds": 0.0,
            "sdk_execution_seconds": 0.0,
            "finalization_seconds": 0.0,
        }
        setup_started = self.clock()
        if deadline is not None:
            deadline.check("sample setup")
        if not transcript.strip():
            raise ValueError("an explicit non-empty debug transcript is required")
        sample = self.dataset.get_sample(sample_id)
        catalog = self.dataset.load_catalog()
        allowed_tools = catalog.restricted_to(sample.allowed_tool_names)
        phase_timings["sample_setup_seconds"] += self.clock() - setup_started
        if deadline is not None:
            deadline.check("ToolEnv creation")

        # This is the sole ToolEnv construction for this inference.
        toolenv_started = self.clock()
        session = self.session_factory.create(self.dataset.split, sample.id, self.dataset.dataset_root)
        if session_observer is not None:
            session_observer(session)
        self._verify_permissions(sample, session)
        phase_timings["toolenv_creation_seconds"] += self.clock() - toolenv_started

        steps: list[StepResult] = []
        calls_sent = 0
        dispatched_writes: dict[str, str] = {}
        write_dispatch_counts: dict[str, int] = {}
        dispatched_reads: dict[str, Mapping[str, Any]] = {}
        read_dispatch_counts: dict[str, int] = {}
        correction_pending: str | None = None
        corrections_used: set[str] = set()
        identical_action_allowance = _identical_write_allowance(
            transcript, self.config.max_tool_calls
        )
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
            corrective_reason = correction_pending
            correction_pending = None
            decision_started = self.clock()
            try:
                action = decisions.next_action(context)
            except BadDecisionOutput as exc:
                steps.append(StepResult(index, None, "model_bad_output", None, None, str(exc)))
                if corrective_reason == "duplicate_write":
                    termination = "duplicate_recovery_exhausted"
                    break
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
            finally:
                phase_timings["decision_seconds"] += self.clock() - decision_started
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
            if action.name == AGENT_FINISH:
                validation = _validate_finish(action.arguments)
                if not validation["ok"]:
                    steps.append(
                        StepResult(
                            index,
                            action,
                            "invalid_finish",
                            MappingProxyType(validation),
                            MappingProxyType(
                                {
                                    "ok": False,
                                    "instruction": (
                                        "Call agent.finish with status done or unsupported "
                                        "and a non-empty reason."
                                    ),
                                }
                            ),
                            str(validation["error"]),
                        )
                    )
                    if corrective_reason == "duplicate_write":
                        termination = "duplicate_recovery_exhausted"
                        break
                    continue
                finish_status = str(action.arguments["status"])
                steps.append(
                    StepResult(
                        index,
                        action,
                        "agent_finished",
                        MappingProxyType(validation),
                        MappingProxyType({"ok": True, "status": finish_status}),
                    )
                )
                termination = "unsupported_request" if finish_status == "unsupported" else "model_completed"
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
                if corrective_reason == "duplicate_write":
                    termination = "duplicate_recovery_exhausted"
                    break
                continue

            arguments = action.as_arguments()
            validation_started = self.clock()
            validation = session.validate(action.name, arguments)
            phase_timings["validation_seconds"] += self.clock() - validation_started
            if not validation.get("ok"):
                steps.append(
                    StepResult(index, action, "schema_rejected", MappingProxyType(dict(validation)), None)
                )
                if corrective_reason == "duplicate_write":
                    termination = "duplicate_recovery_exhausted"
                    break
                continue
            if calls_sent >= self.config.max_tool_calls:
                observation = MappingProxyType(
                    {
                        "ok": False,
                        "tool_call_budget_exhausted": True,
                        "instruction": (
                            "No more official SDK calls are allowed. Do not propose another application tool. "
                            "Call agent.finish with the best status supported by the completed work."
                        ),
                    }
                )
                steps.append(
                    StepResult(
                        index,
                        action,
                        "tool_call_budget_rejected",
                        MappingProxyType(dict(validation)),
                        observation,
                        "official SDK tool-call budget is exhausted",
                    )
                )
                if "tool_budget" in corrections_used or corrective_reason == "tool_budget":
                    termination = "tool_call_budget_exhausted"
                    break
                corrections_used.add("tool_budget")
                correction_pending = "tool_budget"
                continue

            tool_kind = allowed_tools[action.name].get("kind")
            write_key = _action_key(action.name, arguments) if tool_kind == "write" else None
            read_key = _action_key(action.name, arguments) if tool_kind == "read" else None
            dispatched_count = write_dispatch_counts.get(write_key, 0) if write_key is not None else 0
            if (
                write_key is not None
                and write_key in dispatched_writes
                and dispatched_count >= identical_action_allowance
            ):
                prior_outcome = dispatched_writes[write_key]
                observation = MappingProxyType(
                    {
                        "ok": False,
                        "already_executed": prior_outcome == "succeeded",
                        "already_attempted": True,
                        "prior_outcome": prior_outcome,
                        "instruction": (
                            "This write was already attempted or completed. Do not repeat it. "
                            "If the user's request is fully satisfied, call agent.finish."
                        ),
                    }
                )
                steps.append(
                    StepResult(
                        index,
                        action,
                        "duplicate_write_rejected",
                        MappingProxyType(dict(validation)),
                        observation,
                        "an identical write already reached the SDK in this inference",
                    )
                )
                if "duplicate_write" in corrections_used or corrective_reason == "duplicate_write":
                    termination = "duplicate_recovery_exhausted"
                    break
                corrections_used.add("duplicate_write")
                correction_pending = "duplicate_write"
                continue
            read_count = read_dispatch_counts.get(read_key, 0) if read_key is not None else 0
            if (
                read_key is not None
                and read_key in dispatched_reads
                and read_count >= identical_action_allowance
            ):
                observation = MappingProxyType(
                    {
                        "ok": False,
                        "already_read": True,
                        "previous_result_available": True,
                        "instruction": (
                            "This identical read already succeeded and no write has changed the state. "
                            "Use the previous result, change the query, or call agent.finish."
                        ),
                    }
                )
                steps.append(
                    StepResult(
                        index,
                        action,
                        "duplicate_read_rejected",
                        MappingProxyType(dict(validation)),
                        observation,
                        "an identical read already reached the SDK without an intervening write",
                    )
                )
                if read_key in corrections_used or corrective_reason == f"duplicate_read:{read_key}":
                    termination = "read_loop_exhausted"
                    break
                corrections_used.add(read_key)
                correction_pending = f"duplicate_read:{read_key}"
                continue
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
            if write_key is not None:
                dispatched_reads.clear()
                read_dispatch_counts.clear()
                dispatched_writes[write_key] = "uncertain"
                write_dispatch_counts[write_key] = dispatched_count + 1
            execution_started = self.clock()
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
            finally:
                phase_timings["sdk_execution_seconds"] += self.clock() - execution_started
            status = _classify_observation(action.name, allowed_tools[action.name], observation)
            if write_key is not None:
                dispatched_writes[write_key] = "succeeded" if status == "write_succeeded" else status
            if read_key is not None:
                dispatched_reads[read_key] = MappingProxyType(dict(observation))
                read_dispatch_counts[read_key] = read_count + 1
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
            if corrective_reason == "duplicate_write":
                termination = "duplicate_recovery_exhausted"
                break

        finalization_started = self.clock()
        submission = self.capture_submission(sample.id, session=session)
        phase_timings["finalization_seconds"] += self.clock() - finalization_started
        try:
            if deadline is not None:
                deadline.check("sample output finalization")
        except SampleDeadlineExceeded as exc:
            steps.append(StepResult(len(steps), None, "sample_timeout", None, None, str(exc)))
            termination = "sample_timeout"
        phase_timings["agent_total_seconds"] = self.clock() - run_started
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
            model_request_diagnostics=_model_request_diagnostics(decisions),
            phase_timings=MappingProxyType(phase_timings),
        )

    def capture_submission(
        self, sample_id: str, *, session: ToolSession | None = None
    ) -> Mapping[str, Any]:
        """Return an authoritative SDK snapshot, creating an empty session only if needed."""

        sample = self.dataset.get_sample(sample_id)
        active_session = session
        if active_session is None:
            active_session = self.session_factory.create(
                self.dataset.split, sample.id, self.dataset.dataset_root
            )
        self._verify_permissions(sample, active_session)
        return MappingProxyType(
            {
                "id": sample.id,
                "tool_calls": active_session.get_tool_calls(),
                "state_hash": active_session.get_state_hash(),
            }
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


def _identical_write_allowance(transcript: str, maximum: int) -> int:
    folded = transcript.casefold()
    numeric = _NUMERIC_REPEAT.search(folded)
    if numeric is not None:
        return max(1, min(int(numeric.group(1)), maximum))
    for phrase, count in _WORD_REPEATS.items():
        if phrase in folded:
            return min(count, maximum)
    if _REPEAT_ONCE_MORE.search(folded):
        return min(2, maximum)
    return 1


def _validate_finish(arguments: Mapping[str, Any]) -> dict[str, Any]:
    keys = set(arguments)
    if keys != {"status", "reason"}:
        return {"ok": False, "error": "agent.finish requires exactly status and reason"}
    status = arguments.get("status")
    reason = arguments.get("reason")
    if status not in {"done", "unsupported"}:
        return {"ok": False, "error": "agent.finish status must be done or unsupported"}
    if not isinstance(reason, str) or not reason.strip():
        return {"ok": False, "error": "agent.finish reason must be a non-empty string"}
    return {"ok": True, "error": None}


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


def _model_request_diagnostics(decisions: DecisionSource) -> tuple[Mapping[str, Any], ...]:
    diagnostics = getattr(decisions, "request_diagnostics", ())
    if not isinstance(diagnostics, (list, tuple)):
        return ()
    return tuple(MappingProxyType(dict(item)) for item in diagnostics if isinstance(item, Mapping))
