"""Thin adapter around the installed split-specific official SDK."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class SdkImportError(RuntimeError):
    """The official split SDK could not be imported safely."""


@runtime_checkable
class ToolSession(Protocol):
    """Only runtime-safe SDK methods; intentionally excludes get_state()."""

    allowed: set[str]

    def validate(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]: ...

    def get_tool_calls(self) -> list[dict[str, Any]]: ...

    def get_state_hash(self) -> str: ...


class SessionFactory(Protocol):
    def create(self, split: str, sample_id: str, dataset_root: Path) -> ToolSession: ...


class OfficialSessionFactory:
    """Import the SDK beside the selected split and create exactly one ToolEnv per call."""

    def create(self, split: str, sample_id: str, dataset_root: Path) -> ToolSession:
        split_dir = (dataset_root / split).resolve()
        expected_init = split_dir / "zaic_tools" / "__init__.py"
        if not expected_init.is_file():
            raise SdkImportError(f"official SDK not found: {expected_init}")

        existing = sys.modules.get("zaic_tools")
        if existing is not None:
            loaded_file = Path(getattr(existing, "__file__", "")).resolve()
            if loaded_file != expected_init.resolve():
                raise SdkImportError(f"zaic_tools is already loaded from a different split: {loaded_file}")

        path = str(split_dir)
        added = path not in sys.path
        if added:
            sys.path.insert(0, path)
        try:
            module = importlib.import_module("zaic_tools")
        finally:
            if added:
                sys.path.remove(path)

        session = module.ToolEnv(split, sample_id, root=dataset_root)
        if not isinstance(session, ToolSession):
            raise SdkImportError("official ToolEnv does not expose the verified Session interface")
        return session
