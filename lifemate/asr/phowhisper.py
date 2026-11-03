"""PhoWhisper adapter using the official Transformers pipeline interface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..compliance import SampleDeadline
from .common import (
    ASRDevice,
    ASRDtype,
    BackendFactory,
    BaseASRTranscriber,
    TranscriptionError,
    TranscriptionResult,
    audio_duration_seconds,
    resolve_audio_path,
)
from .whisper import TransformersWhisperBackend


@dataclass(frozen=True)
class PhoWhisperSettings:
    checkpoint: str = "vinai/PhoWhisper-medium"
    revision: str | None = None
    backend: str = "phowhisper-transformers"
    device: ASRDevice = "auto"
    dtype: ASRDtype = "auto"
    language: str | None = "vi"
    max_output_tokens: int = 256
    max_audio_seconds: float = 30.0
    preprocessing_version: str = "phowhisper-transformers-path-v1"

    def __post_init__(self) -> None:
        if not self.checkpoint.strip():
            raise ValueError("checkpoint must be non-empty")
        if self.backend != "phowhisper-transformers":
            raise ValueError(f"unsupported PhoWhisper backend: {self.backend}")
        if self.max_output_tokens < 1 or self.max_audio_seconds <= 0:
            raise ValueError("PhoWhisper token and audio-duration limits must be positive")

    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "backend": self.backend,
            "language": self.language,
            "max_output_tokens": self.max_output_tokens,
            "max_audio_seconds": self.max_audio_seconds,
            "dtype": self.dtype,
            "preprocessing_version": self.preprocessing_version,
        }


class PhoWhisperTranscriber(BaseASRTranscriber):
    """Short-clip PhoWhisper transcriber; inputs over the window fail explicitly."""

    inference_name = "PhoWhisper"

    def __init__(
        self,
        settings: PhoWhisperSettings | None = None,
        *,
        allowed_audio_root: Path | str | None = None,
        backend_factory: BackendFactory | None = None,
        clock=None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "allowed_audio_root": allowed_audio_root,
            "backend_factory": backend_factory or TransformersWhisperBackend,
        }
        if clock is not None:
            kwargs["clock"] = clock
        super().__init__(settings or PhoWhisperSettings(), **kwargs)

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        deadline: SampleDeadline | None = None,
    ) -> TranscriptionResult:
        resolved = resolve_audio_path(audio_path, allowed_root=self.allowed_audio_root)
        duration = audio_duration_seconds(resolved)
        if duration > self.settings.max_audio_seconds:
            raise TranscriptionError(
                f"PhoWhisper input {resolved.name} is {duration:.3f}s, above the "
                f"{self.settings.max_audio_seconds:.3f}s no-chunking limit"
            )
        return super().transcribe(resolved, deadline=deadline)
