from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from lifemate.data import DatasetReader


def make_dataset(
    root: Path,
    *,
    installed_tools: list[str] | None = None,
    history: list[dict[str, Any]] | None = None,
    extra_tools: list[dict[str, Any]] | None = None,
) -> DatasetReader:
    installed_tools = installed_tools or ["alarm.list", "alarm.create"]
    split = root / "public_test"
    split.mkdir(parents=True)
    sample = {
        "id": "public_test_1",
        "audio": "audio/public_test_1/public_test_1.wav",
        "context": {
            "current_time": "2026-10-30T10:00:00+07:00",
            "timezone": "Asia/Ho_Chi_Minh",
            "location": {"lat": 0.0, "lng": 0.0},
            "user_profile": {"name": "Test"},
            "installed_apps": {
                "fixture": {"tools": installed_tools},
            },
        },
        "history": history or [],
    }
    (split / "public_test.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")
    tools = [
        {
            "name": "alarm.list",
            "kind": "read",
            "parameters": {"type": "object", "properties": {}},
            "app": "alarm",
        },
        {
            "name": "alarm.create",
            "kind": "write",
            "parameters": {
                "type": "object",
                "properties": {"time": {"type": "string"}},
                "required": ["time"],
            },
            "app": "alarm",
        },
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
            "app": "sys",
        },
        {
            "name": "sms.send",
            "kind": "write",
            "parameters": {"type": "object", "properties": {}},
            "app": "sms",
        },
    ]
    for extra in extra_tools or []:
        tools = [tool for tool in tools if tool.get("name") != extra.get("name")]
        tools.append(extra)
    catalog = {"schema_version": "test", "apps": [{"id": "fixture", "tools": tools}]}
    (split / "public_test_tools_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    return DatasetReader(root)


class FakeSession:
    def __init__(
        self,
        allowed: set[str],
        *,
        validate: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        call: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        state_hash: str = "official-hash",
    ) -> None:
        self.allowed = allowed
        self.validations: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self._validate = validate
        self._call = call
        self._state_hash = state_hash

    def validate(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments or {}
        self.validations.append({"name": name, "arguments": dict(args)})
        if self._validate:
            return self._validate(name, args)
        return {"ok": True, "error": None}

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        args = arguments or {}
        self.calls.append({"name": name, "arguments": dict(args)})
        if self._call:
            return self._call(name, args)
        return {"ok": True}

    def get_tool_calls(self) -> list[dict[str, Any]]:
        return [dict(call) for call in self.calls]

    def get_state_hash(self) -> str:
        return self._state_hash


class FakeFactory:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.creations: list[tuple[str, str, Path]] = []

    def create(self, split: str, sample_id: str, dataset_root: Path) -> FakeSession:
        self.creations.append((split, sample_id, dataset_root))
        return self.session
