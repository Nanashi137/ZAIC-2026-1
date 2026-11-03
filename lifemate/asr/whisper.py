"""OpenAI Whisper adapters for Transformers and faster-whisper runtimes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..compliance import SampleDeadline
from .common import (
    ASRDependencyError,
    ASRDevice,
    ASRDtype,
    ASRInitializationError,
    BackendFactory,
    BaseASRTranscriber,
    TranscriptionError,
    TranscriptionResult,
    audio_duration_seconds,
    model_parameter_count,
    resolve_audio_path,
    resolve_torch_placement,
    transformers_architecture_parameter_count,
)

_FASTER_PARAMETER_SOURCES = {
    "medium": "openai/whisper-medium",
    "systran/faster-whisper-medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "large-v3": "openai/whisper-large-v3",
    "systran/faster-whisper-large-v3": "openai/whisper-large-v3",
}


@dataclass(frozen=True)
class WhisperSettings:
    checkpoint: str = "openai/whisper-medium"
    revision: str | None = None
    backend: str = "whisper-transformers"
    device: ASRDevice = "auto"
    dtype: ASRDtype = "auto"
    language: str | None = "vi"
    max_output_tokens: int = 256
    max_audio_seconds: float = 30.0
    preprocessing_version: str = "whisper-transformers-path-v1"

    def __post_init__(self) -> None:
        _validate_whisper_settings(self, "whisper-transformers")

    def cache_identity(self) -> Mapping[str, Any]:
        return _whisper_cache_identity(self)


@dataclass(frozen=True)
class FasterWhisperSettings:
    checkpoint: str = "medium"
    revision: str | None = None
    parameter_count_checkpoint: str | None = None
    backend: str = "faster-whisper"
    device: ASRDevice = "auto"
    dtype: ASRDtype = "float16"
    language: str | None = "vi"
    max_output_tokens: int = 256
    max_audio_seconds: float = 30.0
    preprocessing_version: str = "faster-whisper-path-v1"

    def __post_init__(self) -> None:
        _validate_whisper_settings(self, "faster-whisper")
        if self.parameter_count_checkpoint is None and self.checkpoint.lower() not in _FASTER_PARAMETER_SOURCES:
            raise ValueError(
                "a custom faster-whisper checkpoint requires parameter_count_checkpoint for exact reporting"
            )
        if self.dtype not in {"auto", "float32", "float16"}:
            raise ValueError("faster-whisper supports auto, float32, or float16 in this benchmark")

    @property
    def canonical_parameter_checkpoint(self) -> str:
        if self.parameter_count_checkpoint is not None:
            return self.parameter_count_checkpoint
        return _FASTER_PARAMETER_SOURCES[self.checkpoint.lower()]

    def cache_identity(self) -> Mapping[str, Any]:
        return {
            **_whisper_cache_identity(self),
            "parameter_count_checkpoint": self.canonical_parameter_checkpoint,
        }


class WhisperTranscriber(BaseASRTranscriber):
    inference_name = "Whisper"

    def __init__(
        self,
        settings: WhisperSettings | None = None,
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
        super().__init__(settings or WhisperSettings(), **kwargs)

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        deadline: SampleDeadline | None = None,
    ) -> TranscriptionResult:
        _enforce_whisper_window(audio_path, self.allowed_audio_root, self.settings.max_audio_seconds, "Whisper")
        return super().transcribe(audio_path, deadline=deadline)


class FasterWhisperTranscriber(BaseASRTranscriber):
    inference_name = "faster-whisper"

    def __init__(
        self,
        settings: FasterWhisperSettings | None = None,
        *,
        allowed_audio_root: Path | str | None = None,
        backend_factory: BackendFactory | None = None,
        clock=None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "allowed_audio_root": allowed_audio_root,
            "backend_factory": backend_factory or FasterWhisperBackend,
        }
        if clock is not None:
            kwargs["clock"] = clock
        super().__init__(settings or FasterWhisperSettings(), **kwargs)

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        deadline: SampleDeadline | None = None,
    ) -> TranscriptionResult:
        _enforce_whisper_window(
            audio_path,
            self.allowed_audio_root,
            self.settings.max_audio_seconds,
            "faster-whisper",
        )
        return super().transcribe(audio_path, deadline=deadline)


class TransformersWhisperBackend:
    """Shared Transformers implementation for Whisper-family checkpoints."""

    def __init__(self, settings: Any) -> None:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
        except ImportError as exc:
            raise ASRDependencyError(
                "Whisper requires torch and transformers; install the ASR benchmark dependencies"
            ) from exc
        device, dtype = resolve_torch_placement(torch, settings)
        model_kwargs: dict[str, Any] = {"dtype": dtype, "low_cpu_mem_usage": True}
        if settings.revision is not None:
            model_kwargs["revision"] = settings.revision
        try:
            self.processor = AutoProcessor.from_pretrained(settings.checkpoint, revision=settings.revision)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(settings.checkpoint, **model_kwargs)
            self.model.to(device)
            self.model.eval()
            self.pipeline = pipeline(
                "automatic-speech-recognition",
                model=self.model,
                tokenizer=self.processor.tokenizer,
                feature_extractor=self.processor.feature_extractor,
                device=device,
                dtype=dtype,
            )
        except Exception as exc:
            raise ASRInitializationError(f"failed to load {settings.checkpoint}: {exc}") from exc
        self.language = settings.language
        self.max_output_tokens = settings.max_output_tokens
        self.parameter_count = model_parameter_count(self.model)
        self.parameter_count_source = "loaded_transformers_model.parameters()"

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        *,
        deadline: SampleDeadline | None = None,
    ) -> tuple[str, str | None]:
        if deadline is not None:
            deadline.check("Whisper inference")
        generate_kwargs: dict[str, Any] = {
            "task": "transcribe",
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": self.max_output_tokens,
        }
        if language is not None:
            generate_kwargs["language"] = language
        output = self.pipeline(
            str(audio_path),
            generate_kwargs=generate_kwargs,
            return_timestamps=False,
        )
        if deadline is not None:
            deadline.check("Whisper inference")
        if not isinstance(output, Mapping) or not isinstance(output.get("text"), str):
            raise TranscriptionError("Whisper pipeline returned an unexpected result")
        return output["text"], language or self.language


class FasterWhisperBackend:
    def __init__(self, settings: FasterWhisperSettings) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRDependencyError(
                "faster-whisper is not installed; install the ASR benchmark dependencies"
            ) from exc
        device, device_index = _faster_device(settings.device)
        compute_type = "default" if settings.dtype == "auto" else settings.dtype
        try:
            self.model = WhisperModel(
                settings.checkpoint,
                device=device,
                device_index=device_index,
                compute_type=compute_type,
                revision=settings.revision,
            )
            self.parameter_count = transformers_architecture_parameter_count(
                settings.canonical_parameter_checkpoint
            )
        except Exception as exc:
            raise ASRInitializationError(f"failed to load {settings.checkpoint}: {exc}") from exc
        self.settings = settings
        self.parameter_count_source = (
            f"tied_transformers_architecture:{settings.canonical_parameter_checkpoint}"
        )

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        *,
        deadline: SampleDeadline | None = None,
    ) -> tuple[str, str | None]:
        if deadline is not None:
            deadline.check("faster-whisper inference")
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            task="transcribe",
            beam_size=1,
            best_of=1,
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            vad_filter=False,
            max_new_tokens=self.settings.max_output_tokens,
        )
        text = " ".join(segment.text.strip() for segment in segments if segment.text.strip())
        if deadline is not None:
            deadline.check("faster-whisper inference")
        detected = getattr(info, "language", None)
        return text, detected if isinstance(detected, str) and detected else language


def _validate_whisper_settings(settings: Any, expected_backend: str) -> None:
    if not settings.checkpoint.strip():
        raise ValueError("checkpoint must be non-empty")
    if settings.backend != expected_backend:
        raise ValueError(f"unsupported Whisper backend: {settings.backend}")
    if settings.max_output_tokens < 1 or settings.max_audio_seconds <= 0:
        raise ValueError("Whisper token and audio-duration limits must be positive")


def _whisper_cache_identity(settings: Any) -> dict[str, Any]:
    return {
        "checkpoint": settings.checkpoint,
        "revision": settings.revision,
        "backend": settings.backend,
        "language": settings.language,
        "max_output_tokens": settings.max_output_tokens,
        "max_audio_seconds": settings.max_audio_seconds,
        "dtype": settings.dtype,
        "preprocessing_version": settings.preprocessing_version,
    }


def _enforce_whisper_window(
    audio_path: Path | str,
    allowed_root: Path | str | None,
    max_audio_seconds: float,
    name: str,
) -> None:
    resolved = resolve_audio_path(audio_path, allowed_root=allowed_root)
    duration = audio_duration_seconds(resolved)
    if duration > max_audio_seconds:
        raise TranscriptionError(
            f"{name} input {resolved.name} is {duration:.3f}s, above the "
            f"{max_audio_seconds:.3f}s no-chunking limit"
        )


def _faster_device(device: str) -> tuple[str, int]:
    if device == "auto":
        return "auto", 0
    if device == "cuda":
        return "cuda", 0
    if device.startswith("cuda:"):
        try:
            return "cuda", int(device.split(":", 1)[1])
        except ValueError as exc:
            raise ASRInitializationError(f"invalid faster-whisper CUDA device: {device}") from exc
    if device == "cpu":
        return "cpu", 0
    raise ASRInitializationError(f"unsupported faster-whisper device: {device}")
