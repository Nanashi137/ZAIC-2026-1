"""LLM context construction, dynamic tools, and sequential model decisions."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .compliance import SampleDeadline
from .model import InferenceSettings, ModelClient, ModelClientError, ModelRequest, ModelResponse, ModelUsage
from .runtime import (
    AGENT_FINISH,
    BadDecisionOutput,
    DecisionContext,
    DecisionModelError,
    DecisionRefusal,
    DecisionRetryLimit,
    StepResult,
    ToolAction,
)
from .tool_schema import ToolSchemaCompatibilityError, adapt_function_parameters

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_FINISH_PARAMETERS: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["done", "unsupported"]},
        "reason": {"type": "string"},
    },
    "required": ["status", "reason"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ModelToolRegistry:
    definitions: tuple[Mapping[str, Any], ...]
    model_to_official: Mapping[str, str]
    official_to_model: Mapping[str, str]

    @classmethod
    def from_allowed(cls, allowed: Mapping[str, Mapping[str, Any]]) -> ModelToolRegistry:
        model_to_official: dict[str, str] = {}
        official_to_model: dict[str, str] = {}
        definitions: list[Mapping[str, Any]] = []
        for official_name in sorted(allowed):
            spec = allowed[official_name]
            model_name = _unique_model_name(official_name, model_to_official)
            model_to_official[model_name] = official_name
            official_to_model[official_name] = model_name
            try:
                parameters = adapt_function_parameters(spec.get("parameters"))
            except ToolSchemaCompatibilityError as exc:
                raise ToolSchemaCompatibilityError(f"tool {official_name!r}: {exc}") from exc
            description = _tool_description(official_name, spec)
            definitions.append(
                {
                    "type": "function",
                    "function": {
                        "name": model_name,
                        "description": description,
                        "parameters": parameters,
                        # Official schemas have optional properties and SDK-specific extensions,
                        # so strict provider mode would change their semantics.
                        "strict": False,
                    },
                }
            )
        finish_model_name = _unique_model_name(AGENT_FINISH, model_to_official)
        model_to_official[finish_model_name] = AGENT_FINISH
        official_to_model[AGENT_FINISH] = finish_model_name
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": finish_model_name,
                    "description": (
                        "Internal controller action agent.finish. Call it immediately when no further official "
                        "tool calls are required. It is never sent to the application SDK."
                    ),
                    "parameters": _FINISH_PARAMETERS,
                    "strict": True,
                },
            }
        )
        return cls(tuple(definitions), model_to_official, official_to_model)


class ContextBuilder:
    """Build model messages from non-privileged sample inputs and runtime observations."""

    def initial_messages(
        self,
        context: DecisionContext,
        registry: ModelToolRegistry,
    ) -> list[dict[str, Any]]:
        sample = context.sample
        public_context = {
            "current_time": sample.context.get("current_time"),
            "timezone": sample.context.get("timezone"),
            "location": _jsonable(sample.context.get("location")),
            "user_profile": _jsonable(sample.context.get("user_profile")),
        }
        finish_model_name = registry.official_to_model[AGENT_FINISH]
        history_source = context.conversation_history if context.conversation_history is not None else sample.history
        history = [_history_turn(turn) for turn in history_source]
        system = (
            "You are a Vietnamese voice-to-action planner for the LifeMate competition. "
            "Choose only from the provided functions. "
            "Prior history tool effects are already applied: never replay them. "
            "Use read tools before writes whenever an ID/reference must come from app data. Never invent IDs. "
            "After every function call, inspect the actual observation before deciding the next step. "
            "An observation with ok=true and done=false is a no-op, not success. "
            "After a write succeeds, do not repeat that operation unless the user explicitly requested another "
            "distinct execution. A subsequent read does not invalidate an earlier successful write. "
            f"When all requested actions are complete, immediately call {finish_model_name} "
            "(the internal action agent.finish). If more actions are required, "
            "continue with those actions. Do not produce free-text JSON to terminate the task. "
            "If required user information is missing, call the provided sys.ask_missing_info function "
            "with exact schema "
            "parameter names. If no available official tool can satisfy the request, call agent.finish with "
            "status unsupported. Never claim an action succeeded unless confirmed by the SDK. "
            "Do not reveal chain-of-thought; make only the next tool decision or the final status."
        )
        user_payload = {
            "public_context": public_context,
            "history": history,
            "final_user_transcript": context.transcript,
            "final_user_transcript_source": (
                "qwen3_asr" if context.conversation_history is not None else "debug_override"
            ),
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))},
        ]


class LlmDecisionSource:
    """One-model-turn-at-a-time decision source with actual observation feedback."""

    def __init__(
        self,
        client: ModelClient,
        settings: InferenceSettings,
        *,
        context_builder: ContextBuilder | None = None,
        max_invalid_proposals: int = 2,
        deadline: SampleDeadline | None = None,
        capture_raw_responses: bool = False,
        minimum_model_time_seconds: float = 0.25,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if max_invalid_proposals < 0:
            raise ValueError("max_invalid_proposals must be non-negative")
        if minimum_model_time_seconds < 0:
            raise ValueError("minimum_model_time_seconds must be non-negative")
        self.client = client
        self.settings = settings
        self.context_builder = context_builder or ContextBuilder()
        self.max_invalid_proposals = max_invalid_proposals
        self.name = f"llm:{client.name}:{settings.model}"
        self.finish_reason = "model_completed"
        self._messages: list[dict[str, Any]] | None = None
        self._registry: ModelToolRegistry | None = None
        self._reported_steps = 0
        self._invalid_proposals = 0
        self._usage = ModelUsage(requests=0)
        self._clock = clock
        self._latency_seconds = 0.0
        self._deadline = deadline
        self._capture_raw_responses = capture_raw_responses
        self._minimum_model_time_seconds = minimum_model_time_seconds
        self._request_diagnostics: list[dict[str, Any]] = []
        self._active_diagnostic: dict[str, Any] | None = None

    @property
    def usage(self) -> Mapping[str, int]:
        return self._usage.as_dict()

    @property
    def latency_seconds(self) -> float:
        return self._latency_seconds

    @property
    def request_diagnostics(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(dict(item) for item in self._request_diagnostics)

    def set_deadline(self, deadline: SampleDeadline) -> None:
        """Attach the batch-created deadline before the first model turn."""

        if self._messages is not None:
            raise RuntimeError("cannot replace deadline after model conversation starts")
        self._deadline = deadline

    def next_action(self, context: DecisionContext) -> ToolAction | None:
        if self._deadline is not None:
            self._deadline.require_duration(self._minimum_model_time_seconds, "agent model step")
        if self._messages is None:
            self._registry = ModelToolRegistry.from_allowed(context.allowed_tools)
            self._messages = self.context_builder.initial_messages(context, self._registry)
        self._append_new_step_feedback(context.steps)
        assert self._registry is not None
        remaining_before = self._deadline.remaining_seconds() if self._deadline is not None else None
        diagnostic: dict[str, Any] = {
            "sample_id": context.sample.id,
            "step_index": len(context.steps),
            "remaining_seconds_before": remaining_before,
        }
        self._request_diagnostics.append(diagnostic)
        self._active_diagnostic = diagnostic
        started = self._clock()
        try:
            response = self.client.complete(
                ModelRequest(
                    messages=tuple(self._messages),
                    tools=self._registry.definitions,
                    settings=self.settings,
                    deadline=self._deadline,
                )
            )
        except ModelClientError as exc:
            diagnostic.update({"response_type": "model_error", "validation_failure": type(exc).__name__})
            raise DecisionModelError(str(exc)) from exc
        finally:
            self._latency_seconds += self._clock() - started
            diagnostic["remaining_seconds_after"] = (
                self._deadline.remaining_seconds() if self._deadline is not None else None
            )
        self._usage = self._usage + response.usage
        diagnostic.update(
            {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.total_tokens,
                "response_type": _response_type(response),
                "selected_tools": [call.name for call in response.tool_calls],
            }
        )
        try:
            return self._parse_response(response)
        finally:
            self._active_diagnostic = None

    def _parse_response(self, response: ModelResponse) -> ToolAction | None:
        assert self._messages is not None and self._registry is not None
        if response.refusal:
            self._record_invalid("refusal", response)
            raise DecisionRefusal(response.refusal)
        if len(response.tool_calls) > 1:
            self._record_invalid("multiple_tool_calls", response)
            self._bad_output(
                "model returned multiple tool calls; none were executed. Sequential mode requires exactly one "
                "next action, and agent.finish must be emitted only after all official operations complete"
            )
        if response.tool_calls:
            call = response.tool_calls[0]
            try:
                arguments = json.loads(call.arguments)
            except json.JSONDecodeError as exc:
                self._record_invalid("invalid_tool_arguments_json", response)
                self._bad_output(f"tool arguments are invalid JSON: {exc.msg}")
                raise AssertionError("unreachable") from exc
            if not isinstance(arguments, dict):
                self._record_invalid("tool_arguments_not_object", response)
                self._bad_output("tool arguments must decode to an object")
            self._invalid_proposals = 0
            self._messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": call.arguments},
                        }
                    ],
                }
            )
            official_name = self._registry.model_to_official.get(call.name, call.name)
            if self._active_diagnostic is not None:
                self._active_diagnostic["selected_tool"] = official_name
            return ToolAction(
                official_name,
                arguments,
                provider_call_id=call.id,
                model_name=call.name,
            )

        self._record_invalid("missing_tool_call", response)
        self._bad_output("model returned no tool call; use agent.finish to terminate")
        raise AssertionError("unreachable")

    def _append_new_step_feedback(self, steps: Sequence[StepResult]) -> None:
        assert self._messages is not None
        for step in steps[self._reported_steps :]:
            feedback: dict[str, Any] = {"status": step.status}
            if step.validation is not None and not step.validation.get("ok"):
                feedback["validation"] = _jsonable(step.validation)
            if step.observation is not None:
                feedback["observation"] = _jsonable(step.observation)
            if step.error:
                feedback["error"] = step.error
            if step.action is not None and step.action.provider_call_id:
                self._messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": step.action.provider_call_id,
                        "content": json.dumps(feedback, ensure_ascii=False, separators=(",", ":")),
                    }
                )
            else:
                self._messages.append(
                    {
                        "role": "user",
                        "content": "Previous proposal feedback: "
                        + json.dumps(feedback, ensure_ascii=False, separators=(",", ":")),
                    }
                )
        self._reported_steps = len(steps)

    def _record_invalid(self, category: str, response: ModelResponse) -> None:
        assert self._messages is not None
        if self._active_diagnostic is not None:
            self._active_diagnostic["validation_failure"] = category
            if self._capture_raw_responses:
                self._active_diagnostic["raw_model_response"] = _diagnostic_response(response)
        if response.content:
            self._messages.append({"role": "assistant", "content": response.content})

    def _bad_output(self, message: str) -> None:
        self._invalid_proposals += 1
        if self._invalid_proposals > self.max_invalid_proposals:
            raise DecisionRetryLimit(message)
        raise BadDecisionOutput(message)


def _unique_model_name(official_name: str, used: Mapping[str, str]) -> str:
    candidate = _SAFE_NAME.sub("__", official_name).strip("_") or "tool"
    if len(candidate) > 64:
        digest = hashlib.sha256(official_name.encode("utf-8")).hexdigest()[:8]
        candidate = f"{candidate[:55]}_{digest}"
    if candidate not in used:
        return candidate
    digest = hashlib.sha256(official_name.encode("utf-8")).hexdigest()[:8]
    return f"{candidate[:55]}_{digest}"


def _tool_description(official_name: str, spec: Mapping[str, Any]) -> str:
    parts = [f"Official tool: {official_name}. Kind: {spec.get('kind', 'unknown')}."]
    if spec.get("description"):
        parts.append(str(spec["description"]))
    parameters = spec.get("parameters")
    if isinstance(parameters, Mapping) and parameters.get("x-one-of"):
        groups = [" + ".join(group) for group in parameters["x-one-of"] if isinstance(group, list)]
        parts.append("At least one complete argument group is required: " + " OR ".join(groups) + ".")
    return " ".join(parts)


def _history_turn(turn: Mapping[str, Any]) -> dict[str, Any]:
    role = turn.get("role")
    if role == "user":
        transcript = turn.get("transcript")
        if isinstance(transcript, str) and transcript.strip():
            return {
                "role": "user",
                "transcript": transcript,
                "language": turn.get("language"),
            }
        return {
            "role": "user",
            "transcript": None,
            "note": "Earlier user audio exists, but no transcript is available in Phase 02.",
        }
    return {
        "role": role,
        "text": turn.get("text"),
        "tool_calls_already_applied": _jsonable(turn.get("tool_calls")),
        "tool_results": _jsonable(turn.get("tool_results")),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _response_type(response: ModelResponse) -> str:
    if response.refusal:
        return "refusal"
    if response.tool_calls:
        return "tool_calls"
    if response.content:
        return "text"
    return "empty"


def _diagnostic_response(response: ModelResponse) -> dict[str, Any]:
    return {
        "content": response.content,
        "refusal": response.refusal,
        "tool_calls": [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in response.tool_calls
        ],
    }
