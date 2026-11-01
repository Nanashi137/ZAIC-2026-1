from __future__ import annotations

import json
from typing import Any

import pytest

from lifemate.agent import LlmDecisionSource, ModelToolRegistry
from lifemate.model import (
    InferenceSettings,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    ScriptedModelClient,
)
from lifemate.runtime import Runtime, RuntimeConfig
from lifemate.tool_schema import (
    FORBIDDEN_ROOT_KEYWORDS,
    ToolSchemaCompatibilityError,
    adapt_function_parameters,
)
from tests.support import FakeFactory, FakeSession, make_dataset


def _tool_response(call_id: str, name: str, arguments: dict[str, Any]) -> ModelResponse:
    return ModelResponse(
        tool_calls=(ModelToolCall(call_id, name, json.dumps(arguments)),),
        usage=ModelUsage(10, 3, 13),
    )


def _source(*responses: ModelResponse, max_invalid_proposals: int = 2):
    client = ScriptedModelClient(list(responses))
    source = LlmDecisionSource(
        client,
        InferenceSettings(model="fixture-model"),
        max_invalid_proposals=max_invalid_proposals,
    )
    return source, client


def test_dynamic_tools_keep_only_allowed_and_translate_sdk_extensions() -> None:
    allowed = {
        "sms.send": {
            "name": "sms.send",
            "kind": "write",
            "description": "Send SMS",
            "parameters": {
                "type": "object",
                "properties": {
                    "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                    "phone_number": {"type": "string"},
                    "content": {"type": "string", "x-match": "fuzzy"},
                },
                "required": ["content"],
                "x-one-of": [["contact_id"], ["phone_number"]],
            },
        }
    }

    registry = ModelToolRegistry.from_allowed(allowed)

    assert registry.model_to_official == {"sms__send": "sms.send"}
    function = registry.definitions[0]["function"]
    assert function["strict"] is False
    assert function["parameters"]["additionalProperties"] is False
    assert not FORBIDDEN_ROOT_KEYWORDS.intersection(function["parameters"])
    assert "contact_id OR phone_number" in function["parameters"]["description"]
    assert "never guess" in function["parameters"]["properties"]["contact_id"]["description"]
    assert "preserve" in function["parameters"]["properties"]["content"]["description"]


def test_schema_adapter_projects_root_one_of_and_any_of_without_wrapping_arguments() -> None:
    one_of = adapt_function_parameters(
        {
            "$defs": {
                "email_variant": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string", "format": "email"},
                        "profile": {
                            "type": "object",
                            "properties": {"nickname": {"type": "string", "nullable": True}},
                            "additionalProperties": False,
                        },
                    },
                    "required": ["email"],
                }
            },
            "oneOf": [
                {"$ref": "#/$defs/email_variant"},
                {
                    "type": "object",
                    "properties": {"phone": {"type": "string"}},
                    "required": ["phone"],
                },
            ],
        }
    )

    assert one_of["type"] == "object"
    assert set(one_of["properties"]) == {"email", "profile", "phone"}
    assert not FORBIDDEN_ROOT_KEYWORDS.intersection(one_of)
    assert one_of["properties"]["profile"]["properties"]["nickname"]["type"] == ["string", "null"]
    assert "Exactly one argument variant" in one_of["description"]

    any_of = adapt_function_parameters(
        {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
                {
                    "type": "object",
                    "properties": {
                        "ids": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"id": {"type": "string"}},
                                "required": ["id"],
                            },
                        }
                    },
                    "required": ["ids"],
                },
            ]
        }
    )

    assert set(any_of["properties"]) == {"query", "ids"}
    assert not FORBIDDEN_ROOT_KEYWORDS.intersection(any_of)
    assert any_of["properties"]["ids"]["items"]["properties"]["id"]["type"] == "string"
    assert "At least one argument variant" in any_of["description"]


def test_schema_adapter_preserves_all_of_constraints_and_rejects_unsafe_root_constraints() -> None:
    adapted = adapt_function_parameters(
        {
            "allOf": [
                {
                    "type": "object",
                    "properties": {"contact_id": {"type": "string"}},
                    "required": ["contact_id"],
                },
                {
                    "type": "object",
                    "properties": {"email": {"type": "string"}},
                    "required": ["email"],
                },
            ]
        }
    )

    assert adapted["required"] == ["contact_id", "email"]
    assert not FORBIDDEN_ROOT_KEYWORDS.intersection(adapted)
    for keyword in ("enum", "const", "not"):
        with pytest.raises(ToolSchemaCompatibilityError, match=f"root-level {keyword}"):
            adapt_function_parameters({"type": "object", keyword: [] if keyword == "enum" else {}})


def test_model_loop_uses_observation_id_for_dependent_read_then_write(tmp_path) -> None:
    contact_tool = {
        "name": "contacts.search",
        "kind": "read",
        "description": "Search contacts",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        "app": "contacts",
    }
    sms_tool = {
        "name": "sms.send",
        "kind": "write",
        "description": "Send SMS",
        "parameters": {
            "type": "object",
            "properties": {
                "contact_id": {"type": "string", "x-ref": "contacts.contacts"},
                "content": {"type": "string"},
            },
            "required": ["contact_id", "content"],
        },
        "app": "sms",
    }
    installed = ["contacts.search", "sms.send"]
    allowed = {*installed, "sys.ask_missing_info"}

    def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "contacts.search":
            return {"ok": True, "contacts": [{"id": "observed_contact_7", "name": "Lan"}]}
        return {"ok": True, "sent_to": arguments["contact_id"]}

    session = FakeSession(allowed, call=call)
    history = [
        {"role": "user", "audio": "audio/private-earlier.wav"},
        {
            "role": "assistant",
            "text": "Đã kiểm tra",
            "tool_calls": [{"name": "contacts.search", "arguments": {"query": "Lan"}}],
            "tool_results": [{"ok": True}],
        },
    ]
    dataset = make_dataset(
        tmp_path,
        installed_tools=installed,
        extra_tools=[contact_tool, sms_tool],
        history=history,
    )
    source, client = _source(
        _tool_response("call_read", "contacts__search", {"query": "Lan"}),
        _tool_response(
            "call_write",
            "sms__send",
            {"contact_id": "observed_contact_7", "content": "Xin chào"},
        ),
        ModelResponse(content='{"status":"done","reason":"message sent"}', usage=ModelUsage(5, 2, 7)),
    )

    result = Runtime(dataset, session_factory=FakeFactory(session)).run(
        "public_test_1", "nhắn Lan là xin chào", source
    )

    assert [call["name"] for call in result.submission["tool_calls"]] == ["contacts.search", "sms.send"]
    assert result.submission["tool_calls"][1]["arguments"]["contact_id"] == "observed_contact_7"
    assert result.termination_reason == "model_completed"
    assert result.model_usage == {
        "input_tokens": 25,
        "output_tokens": 8,
        "total_tokens": 33,
        "requests": 3,
    }
    second_request_messages = client.requests[1].messages
    tool_feedback = [message for message in second_request_messages if message["role"] == "tool"]
    assert "observed_contact_7" in tool_feedback[-1]["content"]
    offered = {
        definition["function"]["name"]
        for definition in client.requests[0].tools
    }
    assert offered == {"contacts__search", "sms__send", "sys__ask_missing_info"}
    initial_payload = json.loads(client.requests[0].messages[1]["content"])
    assert initial_payload["public_context"]["current_time"] == "2026-10-30T10:00:00+07:00"
    assert initial_payload["final_user_transcript_source"] == "debug_override"
    assert initial_payload["history"][1]["tool_calls_already_applied"][0]["name"] == "contacts.search"
    assert "private-earlier.wav" not in client.requests[0].messages[1]["content"]


def test_missing_information_calls_verified_system_tool_and_stops(tmp_path) -> None:
    allowed = {"alarm.list", "alarm.create", "sys.ask_missing_info"}
    session = FakeSession(allowed, call=lambda _name, _args: {"ok": True, "waiting_for_user": True})
    dataset = make_dataset(tmp_path)
    source, client = _source(
        _tool_response(
            "call_missing",
            "sys__ask_missing_info",
            {"tool_name": "alarm.create", "params": ["time"]},
        )
    )

    result = Runtime(dataset, session_factory=FakeFactory(session)).run(
        "public_test_1", "đặt báo thức", source
    )

    assert result.termination_reason == "awaiting_user"
    assert result.steps[0].status == "meta_succeeded"
    assert len(client.requests) == 1


def test_unavailable_tool_is_not_executed_and_model_can_finish_unsupported(tmp_path) -> None:
    allowed = {"alarm.list", "alarm.create", "sys.ask_missing_info"}
    session = FakeSession(allowed)
    dataset = make_dataset(tmp_path)
    source, _ = _source(
        _tool_response("call_unknown", "sms__send", {"content": "hello"}),
        ModelResponse(content='{"status":"unsupported","reason":"SMS is unavailable"}'),
    )

    result = Runtime(dataset, session_factory=FakeFactory(session)).run("public_test_1", "send SMS", source)

    assert result.steps[0].status == "permission_rejected"
    assert session.calls == []
    assert result.termination_reason == "unsupported_request"


def test_bad_model_arguments_retry_without_sdk_execution(tmp_path) -> None:
    allowed = {"alarm.list", "alarm.create", "sys.ask_missing_info"}
    session = FakeSession(allowed)
    dataset = make_dataset(tmp_path)
    source, client = _source(
        ModelResponse(tool_calls=(ModelToolCall("bad", "alarm__list", "{not json"),)),
        _tool_response("good", "alarm__list", {}),
        ModelResponse(content='{"status":"done","reason":"listed"}'),
    )

    result = Runtime(dataset, session_factory=FakeFactory(session)).run("public_test_1", "list alarms", source)

    assert [step.status for step in result.steps] == ["model_bad_output", "read_succeeded"]
    assert session.calls == [{"name": "alarm.list", "arguments": {}}]
    assert len(client.requests) == 3


def test_x_one_of_rejection_is_fed_back_then_model_requests_missing_info(tmp_path) -> None:
    sms_tool = {
        "name": "sms.send",
        "kind": "write",
        "parameters": {
            "type": "object",
            "properties": {
                "contact_id": {"type": "string"},
                "phone_number": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["content"],
            "x-one-of": [["contact_id"], ["phone_number"]],
        },
        "app": "sms",
    }
    installed = ["sms.send"]
    allowed = {"sms.send", "sys.ask_missing_info"}

    def validate(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "sms.send" and not (arguments.get("contact_id") or arguments.get("phone_number")):
            return {"ok": False, "error": "need contact_id or phone_number"}
        return {"ok": True, "error": None}

    def call(name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        assert name == "sys.ask_missing_info"
        return {"ok": True, "waiting_for_user": True}

    session = FakeSession(allowed, validate=validate, call=call)
    dataset = make_dataset(tmp_path, installed_tools=installed, extra_tools=[sms_tool])
    source, client = _source(
        _tool_response("invalid_sms", "sms__send", {"content": "hello"}),
        _tool_response(
            "need_contact",
            "sys__ask_missing_info",
            {"tool_name": "sms.send", "params": ["contact_id"]},
        ),
    )

    result = Runtime(dataset, session_factory=FakeFactory(session)).run("public_test_1", "send hello", source)

    assert [step.status for step in result.steps] == ["schema_rejected", "meta_succeeded"]
    assert [call["name"] for call in session.calls] == ["sys.ask_missing_info"]
    validation_feedback = [message for message in client.requests[1].messages if message["role"] == "tool"]
    assert "need contact_id" in validation_feedback[-1]["content"]


def test_refusal_model_error_bad_output_limit_and_max_steps_terminate_safely(tmp_path) -> None:
    allowed = {"alarm.list", "alarm.create", "sys.ask_missing_info"}
    dataset = make_dataset(tmp_path)

    refused, _ = _source(ModelResponse(refusal="cannot comply"))
    refusal_result = Runtime(dataset, session_factory=FakeFactory(FakeSession(allowed))).run(
        "public_test_1", "list", refused
    )
    assert refusal_result.termination_reason == "model_refusal"
    assert refusal_result.submission["tool_calls"] == []

    exhausted_client = ScriptedModelClient([])
    failed = LlmDecisionSource(exhausted_client, InferenceSettings(model="fixture"))
    error_result = Runtime(dataset, session_factory=FakeFactory(FakeSession(allowed))).run(
        "public_test_1", "list", failed
    )
    assert error_result.termination_reason == "model_error"

    limited, _ = _source(
        ModelResponse(content="bad"),
        ModelResponse(content="still bad"),
        max_invalid_proposals=1,
    )
    limit_result = Runtime(dataset, session_factory=FakeFactory(FakeSession(allowed))).run(
        "public_test_1", "list", limited
    )
    assert limit_result.termination_reason == "model_bad_output_limit"

    capped, client = _source(
        _tool_response("one", "alarm__list", {}),
        _tool_response("two", "alarm__list", {}),
        ModelResponse(content='{"status":"done","reason":"unused"}'),
    )
    cap_result = Runtime(
        dataset,
        session_factory=FakeFactory(FakeSession(allowed)),
        config=RuntimeConfig(max_steps=2, max_tool_calls=6),
    ).run("public_test_1", "list", capped)
    assert cap_result.termination_reason == "max_steps_exhausted"
    assert len(client.requests) == 2


def test_noop_observation_is_returned_to_model_before_finish(tmp_path) -> None:
    update_tool = {
        "name": "alarm.update",
        "kind": "write",
        "parameters": {
            "type": "object",
            "properties": {"alarm_id": {"type": "string", "x-ref": "alarm.alarms"}},
            "required": ["alarm_id"],
        },
        "app": "alarm",
    }
    installed = ["alarm.update"]
    allowed = {"alarm.update", "sys.ask_missing_info"}
    session = FakeSession(
        allowed,
        call=lambda _name, _args: {"ok": True, "done": False, "message": "not found"},
    )
    dataset = make_dataset(tmp_path, installed_tools=installed, extra_tools=[update_tool])
    source, client = _source(
        _tool_response("noop", "alarm__update", {"alarm_id": "observed-but-stale"}),
        ModelResponse(content='{"status":"done","reason":"no matching alarm"}'),
    )

    result = Runtime(dataset, session_factory=FakeFactory(session)).run("public_test_1", "update alarm", source)

    assert result.steps[0].status == "no_op"
    assert result.termination_reason == "model_completed"
    feedback = [message for message in client.requests[1].messages if message["role"] == "tool"][-1]
    assert '"status":"no_op"' in feedback["content"]
