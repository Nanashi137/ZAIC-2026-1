"""Read-only access to challenge inputs and the split tool catalog.

This module never opens audio, environment state, or the out-of-scope train split.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

JsonObject = dict[str, Any]
_SPLIT_NAME = re.compile(r"^[A-Za-z0-9_]+$")
SYSTEM_TOOLS = frozenset({"sys.ask_missing_info"})


class DatasetError(ValueError):
    """The requested split or official input data is invalid."""


@dataclass(frozen=True)
class Sample:
    """One input row. Audio paths are retained as metadata and are not opened."""

    id: str
    audio: str
    context: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]

    @property
    def installed_tool_names(self) -> frozenset[str]:
        installed_apps = self.context.get("installed_apps")
        if not isinstance(installed_apps, dict):
            raise DatasetError(f"sample {self.id!r}: context.installed_apps must be an object")
        names: set[str] = set()
        for app_name, app in installed_apps.items():
            if not isinstance(app_name, str) or not isinstance(app, dict):
                raise DatasetError(f"sample {self.id!r}: malformed installed app entry")
            tools = app.get("tools", [])
            if not isinstance(tools, list) or not all(isinstance(name, str) for name in tools):
                raise DatasetError(f"sample {self.id!r}: installed app {app_name!r} has invalid tools")
            names.update(tools)
        return frozenset(names)

    @property
    def allowed_tool_names(self) -> frozenset[str]:
        """Sample tools plus the SDK's always-available system meta tool."""

        return self.installed_tool_names | SYSTEM_TOOLS


@dataclass(frozen=True)
class Catalog:
    schema_version: str
    tools: Mapping[str, Mapping[str, Any]]

    def restricted_to(self, names: frozenset[str]) -> Mapping[str, Mapping[str, Any]]:
        """Return only catalog entries allowed for this sample."""

        return MappingProxyType({name: self.tools[name] for name in names if name in self.tools})


class DatasetReader:
    """Stream official JSONL and load catalog metadata without touching app state."""

    def __init__(
        self,
        dataset_root: Path | str,
        split: str = "public_test",
        *,
        allow_train: bool = False,
    ) -> None:
        if not _SPLIT_NAME.fullmatch(split):
            raise DatasetError(f"invalid split name: {split!r}")
        if split == "train" and not allow_train:
            raise DatasetError("the train split is explicitly out of scope")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.split_dir = self.dataset_root / split
        self.input_path = self.split_dir / f"{split}.jsonl"
        self.catalog_path = self.split_dir / f"{split}_tools_catalog.json"

    def iter_samples(self) -> Iterator[Sample]:
        if not self.input_path.is_file():
            raise FileNotFoundError(f"split input not found: {self.input_path}")
        with self.input_path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(f"{self.input_path}:{line_number}: invalid JSON") from exc
                yield self._parse_sample(raw, line_number)

    def get_sample(self, sample_id: str) -> Sample:
        for sample in self.iter_samples():
            if sample.id == sample_id:
                return sample
        raise KeyError(f"unknown sample id {sample_id!r} in {self.split}")

    def load_catalog(self) -> Catalog:
        if not self.catalog_path.is_file():
            raise FileNotFoundError(f"tool catalog not found: {self.catalog_path}")
        raw = json.loads(self.catalog_path.read_text(encoding="utf-8-sig"))
        apps = raw.get("apps")
        if not isinstance(apps, list):
            raise DatasetError(f"{self.catalog_path}: apps must be an array")
        tools: dict[str, Mapping[str, Any]] = {}
        for app in apps:
            if not isinstance(app, dict) or not isinstance(app.get("tools"), list):
                raise DatasetError(f"{self.catalog_path}: malformed app entry")
            for tool in app["tools"]:
                if not isinstance(tool, dict) or not isinstance(tool.get("name"), str):
                    raise DatasetError(f"{self.catalog_path}: malformed tool entry")
                name = tool["name"]
                if name in tools:
                    raise DatasetError(f"{self.catalog_path}: duplicate tool {name!r}")
                tools[name] = MappingProxyType(tool)
        return Catalog(str(raw.get("schema_version", "")), MappingProxyType(tools))

    @staticmethod
    def _parse_sample(raw: Any, line_number: int) -> Sample:
        if not isinstance(raw, dict):
            raise DatasetError(f"input line {line_number}: expected an object")
        sample_id = raw.get("id")
        audio = raw.get("audio")
        context = raw.get("context")
        history = raw.get("history")
        if not isinstance(sample_id, str) or not isinstance(audio, str):
            raise DatasetError(f"input line {line_number}: id and audio must be strings")
        if not isinstance(context, dict) or not isinstance(history, list):
            raise DatasetError(f"sample {sample_id!r}: context/history have invalid types")
        if not all(isinstance(turn, dict) for turn in history):
            raise DatasetError(f"sample {sample_id!r}: every history turn must be an object")
        return Sample(
            id=sample_id,
            audio=audio,
            context=MappingProxyType(context),
            history=tuple(MappingProxyType(turn) for turn in history),
        )
