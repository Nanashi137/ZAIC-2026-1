from __future__ import annotations

from typing import Any

from lifemate.runtime import (
    DecisionContext,
    ExplicitActionSource,
    Runtime,
    RuntimeConfig,
    ToolAction,
)
from tests.support import FakeFactory, FakeSession, make_dataset


def _runtime(tmp_path, session: FakeSession, **dataset_options: Any) -> tuple[Runtime, FakeFactory]:
    dataset = make_dataset(tmp_path, **dataset_options)
    factory = FakeFactory(session)
    return Runtime(dataset, session_factory=factory), factory


def test_permission_and_schema_rejections_never_reach_call(tmp_path) -> None:
    allowed = {"alarm.list", "alarm.create", "sys.ask_missing_info"}

    def validate(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        del arguments
        return {"ok": name != "alarm.create", "error": "missing time" if name == "alarm.create" else None}

    session = FakeSession(allowed, validate=validate)
    runtime, factory = _runtime(tmp_path, session)
    result = runtime.run(
        "public_test_1",
        "debug request",
        ExplicitActionSource(
            [
                ToolAction("sms.send", {}),
                ToolAction("alarm.create", {}),
            ]
        ),
    )

    assert [step.status for step in result.steps] == ["permission_rejected", "schema_rejected"]
    assert session.calls == []
    assert [item["name"] for item in session.validations] == ["alarm.create"]
    assert len(factory.creations) == 1


def test_history_calls_are_context_only_and_never_replayed(tmp_path) -> None:
    history = [
        {"role": "user", "audio": "audio/earlier.wav"},
        {
            "role": "assistant",
            "text": "done",
            "tool_calls": [{"name": "alarm.create", "arguments": {"time": "07:00"}}],
            "tool_results": [{"ok": True}],
        },
    ]
    allowed = {"alarm.list", "alarm.create", "sys.ask_missing_info"}
    session = FakeSession(allowed)
    runtime, _ = _runtime(tmp_path, session, history=history)

    result = runtime.run(
        "public_test_1", "list current alarms", ExplicitActionSource([ToolAction("alarm.list", {})])
    )

    assert result.submission["tool_calls"] == [{"name": "alarm.list", "arguments": {}}]
    assert len(session.calls) == 1


def test_sdk_rejection_is_not_filtered_from_official_trace(tmp_path) -> None:
    allowed = {"alarm.list", "alarm.create", "sys.ask_missing_info"}
    session = FakeSession(allowed, call=lambda _name, _args: {"ok": False, "error": "SDK rejected"})
    runtime, _ = _runtime(tmp_path, session)

    result = runtime.run(
        "public_test_1", "list alarms", ExplicitActionSource([ToolAction("alarm.list", {})])
    )

    assert result.steps[0].status == "sdk_rejected"
    assert result.submission["tool_calls"] == [{"name": "alarm.list", "arguments": {}}]


def test_noop_and_executor_exception_are_distinguished(tmp_path) -> None:
    allowed = {"alarm.list", "alarm.create", "sys.ask_missing_info"}
    observations = iter(
        [
            {"ok": True, "done": False, "message": "not found"},
            {"ok": True, "done": False, "message": "ValueError: broken app"},
        ]
    )
    session = FakeSession(allowed, call=lambda _name, _args: next(observations))
    runtime, _ = _runtime(tmp_path, session)
    result = runtime.run(
        "public_test_1",
        "debug",
        ExplicitActionSource(
            [
                ToolAction("alarm.create", {"time": "07:00"}),
                ToolAction("alarm.create", {"time": "08:00"}),
            ]
        ),
    )

    assert [step.status for step in result.steps] == ["no_op", "tool_error"]


def test_clarification_stops_for_user_and_out_of_scope_emits_empty_calls(tmp_path) -> None:
    allowed = {"alarm.list", "alarm.create", "sys.ask_missing_info"}
    session = FakeSession(allowed, call=lambda _name, _args: {"ok": True, "waiting_for_user": True})
    runtime, _ = _runtime(tmp_path, session)
    clarified = runtime.run(
        "public_test_1",
        "make an alarm",
        ExplicitActionSource(
            [ToolAction("sys.ask_missing_info", {"tool_name": "alarm.create", "params": ["time"]})]
        ),
    )
    assert clarified.termination_reason == "awaiting_user"
    assert clarified.steps[0].status == "meta_succeeded"

    empty_session = FakeSession(allowed)
    empty_runtime = Runtime(runtime.dataset, session_factory=FakeFactory(empty_session))
    out_of_scope = empty_runtime.run("public_test_1", "unsupported request", ExplicitActionSource([]))
    assert out_of_scope.submission["tool_calls"] == []
    assert out_of_scope.termination_reason == "decision_source_done"


def test_dependent_source_uses_read_observation_before_write(tmp_path) -> None:
    contact_tool = {
        "name": "contacts.search",
        "kind": "read",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        "app": "contacts",
    }
    installed = ["contacts.search", "sms.send"]
    allowed = {*installed, "sys.ask_missing_info"}

    def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "contacts.search":
            return {"ok": True, "contacts": [{"id": "contact_from_observation"}]}
        return {"ok": True, "sent_to": arguments["contact_id"]}

    class DependentSource:
        name = "dependent_test_source"

        def next_action(self, context: DecisionContext) -> ToolAction | None:
            if not context.steps:
                return ToolAction("contacts.search", {"query": "Lan"})
            if len(context.steps) == 1:
                contact_id = context.steps[0].observation["contacts"][0]["id"]  # type: ignore[index]
                return ToolAction("sms.send", {"contact_id": contact_id, "content": "hello"})
            return None

    session = FakeSession(allowed, call=call)
    runtime, _ = _runtime(
        tmp_path,
        session,
        installed_tools=installed,
        extra_tools=[contact_tool],
    )
    result = runtime.run("public_test_1", "message Lan", DependentSource())

    assert [call["name"] for call in result.submission["tool_calls"]] == ["contacts.search", "sms.send"]
    assert result.submission["tool_calls"][1]["arguments"]["contact_id"] == "contact_from_observation"
    assert result.submission["state_hash"] == "official-hash"


def test_uncertain_write_execution_stops_without_retry_or_new_session(tmp_path) -> None:
    allowed = {"alarm.list", "alarm.create", "sys.ask_missing_info"}

    def uncertain(_name: str, _arguments: dict[str, Any]) -> dict[str, Any]:
        raise TimeoutError("outcome unknown")

    session = FakeSession(allowed, call=uncertain)
    runtime, factory = _runtime(tmp_path, session)
    result = runtime.run(
        "public_test_1",
        "create alarm",
        ExplicitActionSource(
            [
                ToolAction("alarm.create", {"time": "07:00"}),
                ToolAction("alarm.create", {"time": "07:00"}),
            ]
        ),
    )

    assert result.termination_reason == "execution_uncertain"
    assert result.steps[0].status == "execution_uncertain"
    assert session.calls == [{"name": "alarm.create", "arguments": {"time": "07:00"}}]
    assert len(factory.creations) == 1


def test_identical_write_is_rejected_then_corrective_finish_succeeds(tmp_path) -> None:
    allowed = {"alarm.list", "alarm.create", "sys.ask_missing_info"}
    session = FakeSession(allowed)
    runtime, _ = _runtime(tmp_path, session)
    action = ToolAction("alarm.create", {"time": "07:00"})

    result = runtime.run(
        "public_test_1",
        "create alarm",
        ExplicitActionSource(
            [
                action,
                action,
                ToolAction("agent.finish", {"status": "done", "reason": "already created"}),
            ]
        ),
    )

    assert result.termination_reason == "model_completed"
    assert [step.status for step in result.steps] == [
        "write_succeeded",
        "duplicate_write_rejected",
        "agent_finished",
    ]
    assert session.calls == [{"name": "alarm.create", "arguments": {"time": "07:00"}}]


def test_distinct_writes_are_not_deduplicated(tmp_path) -> None:
    session = FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"})
    runtime, _ = _runtime(tmp_path, session)

    result = runtime.run(
        "public_test_1",
        "create two different alarms",
        ExplicitActionSource(
            [
                ToolAction("alarm.create", {"time": "07:00"}),
                ToolAction("alarm.create", {"time": "08:00"}),
            ]
        ),
    )

    assert [step.status for step in result.steps] == ["write_succeeded", "write_succeeded"]
    assert len(session.calls) == 2


def test_explicitly_requested_identical_repeat_is_allowed_twice(tmp_path) -> None:
    session = FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"})
    runtime, _ = _runtime(tmp_path, session)
    action = ToolAction("alarm.create", {"time": "07:00"})

    result = runtime.run(
        "public_test_1",
        "repeat it twice",
        ExplicitActionSource([action, action]),
    )

    assert [step.status for step in result.steps] == ["write_succeeded", "write_succeeded"]
    assert len(session.calls) == 2


def test_identical_read_is_rejected_locally_then_finish_remains_available(tmp_path) -> None:
    session = FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"})
    runtime, _ = _runtime(tmp_path, session)
    read = ToolAction("alarm.list", {})

    result = runtime.run(
        "public_test_1",
        "list alarms",
        ExplicitActionSource(
            [
                read,
                read,
                ToolAction("agent.finish", {"status": "done", "reason": "used first result"}),
            ]
        ),
    )

    assert result.termination_reason == "model_completed"
    assert [step.status for step in result.steps] == [
        "read_succeeded",
        "duplicate_read_rejected",
        "agent_finished",
    ]
    assert session.calls == [{"name": "alarm.list", "arguments": {}}]


def test_repeated_read_after_correction_terminates_loop(tmp_path) -> None:
    session = FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"})
    runtime, _ = _runtime(tmp_path, session)
    read = ToolAction("alarm.list", {})

    result = runtime.run(
        "public_test_1", "list alarms", ExplicitActionSource([read, read, read])
    )

    assert result.termination_reason == "read_loop_exhausted"
    assert len(session.calls) == 1


def test_tool_call_budget_rejection_allows_finish_without_another_sdk_call(tmp_path) -> None:
    session = FakeSession({"alarm.list", "alarm.create", "sys.ask_missing_info"})
    dataset = make_dataset(tmp_path)
    runtime = Runtime(
        dataset,
        session_factory=FakeFactory(session),
        config=RuntimeConfig(max_steps=3, max_tool_calls=1),
    )

    result = runtime.run(
        "public_test_1",
        "list alarms",
        ExplicitActionSource(
            [
                ToolAction("alarm.list", {}),
                ToolAction("alarm.list", {}),
                ToolAction("agent.finish", {"status": "done", "reason": "budget reached"}),
            ]
        ),
    )

    assert result.termination_reason == "model_completed"
    assert [step.status for step in result.steps] == [
        "read_succeeded",
        "tool_call_budget_rejected",
        "agent_finished",
    ]
    assert len(session.calls) == 1
