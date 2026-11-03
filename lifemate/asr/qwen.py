"""Qwen3-ASR adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from ..compliance import SampleDeadline, configure_greedy_generation
from .common import (
    ASRDependencyError,
    ASRDevice,
    ASRDtype,
    ASRInitializationError,
    BackendFactory,
    BaseASRTranscriber,
    TranscriptionError,
    model_parameter_count,
    optional_string,
    resolve_torch_placement,
)

ASRBackendName = Literal["qwen-asr-transformers", "native-hf"]


@dataclass(frozen=True)
class Qwen3ASRSettings:
    checkpoint: str = "Qwen/Qwen3-ASR-0.6B"
    revision: str | None = None
    backend: ASRBackendName = "qwen-asr-transformers"
    device: ASRDevice = "auto"
    dtype: ASRDtype = "auto"
    language: str | None = "Vietnamese"
    max_output_tokens: int = 256
    max_inference_batch_size: int = 1
    preprocessing_version: str = "qwen-direct-path-v1"

    def __post_init__(self) -> None:
        if not self.checkpoint.strip():
            raise ValueError("checkpoint must be non-empty")
        if self.max_output_tokens < 1 or self.max_inference_batch_size < 1:
            raise ValueError("ASR token and batch limits must be positive")
        is_hf_checkpoint = self.checkpoint.lower().endswith("-hf")
        if self.backend == "native-hf" and not is_hf_checkpoint:
            raise ValueError("the native-hf backend requires a Qwen3-ASR *-hf checkpoint")
        if self.backend == "qwen-asr-transformers" and is_hf_checkpoint:
            raise ValueError("the qwen-asr-transformers backend cannot use a *-hf checkpoint")

    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "backend": self.backend,
            "language": self.language,
            "max_output_tokens": self.max_output_tokens,
            "dtype": self.dtype,
            "preprocessing_version": self.preprocessing_version,
        }


class Qwen3ASRTranscriber(BaseASRTranscriber):
    """Lazy, reusable Qwen3-ASR transcriber for local official audio files."""

    inference_name = "Qwen3-ASR"

    def __init__(
        self,
        settings: Qwen3ASRSettings | None = None,
        *,
        allowed_audio_root: Path | str | None = None,
        backend_factory: BackendFactory | None = None,
        clock=None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "allowed_audio_root": allowed_audio_root,
            "backend_factory": backend_factory or _load_backend,
        }
        if clock is not None:
            kwargs["clock"] = clock
        super().__init__(settings or Qwen3ASRSettings(), **kwargs)


def _load_backend(settings: Qwen3ASRSettings):
    if settings.backend == "qwen-asr-transformers":
        return _QwenPackageBackend(settings)
    if settings.backend == "native-hf":
        return _NativeHuggingFaceBackend(settings)
    raise ASRInitializationError(f"unsupported Qwen3-ASR backend: {settings.backend}")


class _QwenPackageBackend:
    def __init__(self, settings: Qwen3ASRSettings) -> None:
        try:
            import torch
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise ASRDependencyError(
                "qwen-asr backend is not installed; install the ASR dependency group"
            ) from exc
        device, dtype = resolve_torch_placement(torch, settings)
        kwargs: dict[str, Any] = {
            "dtype": dtype,
            "device_map": device,
            "max_inference_batch_size": settings.max_inference_batch_size,
            "max_new_tokens": settings.max_output_tokens,
        }
        if settings.revision is not None:
            kwargs["revision"] = settings.revision
        try:
            self.model = Qwen3ASRModel.from_pretrained(settings.checkpoint, **kwargs)
        except Exception as exc:
            raise ASRInitializationError(f"failed to load {settings.checkpoint}: {exc}") from exc
        if not configure_greedy_generation(self.model.model):
            raise ASRInitializationError("qwen-asr Transformers model has no generation_config")
        self.parameter_count = model_parameter_count(self.model.model)
        self.parameter_count_source = "loaded_qwen_transformers_model.parameters()"

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        *,
        deadline: SampleDeadline | None = None,
    ) -> tuple[str, str | None]:
        if deadline is not None:
            deadline.check("Qwen ASR inference")
        results = self.model.transcribe(audio=str(audio_path), language=language)
        if deadline is not None:
            deadline.check("Qwen ASR inference")
        if not results:
            raise TranscriptionError("Qwen3-ASR returned no result")
        first = results[0]
        return first.text, getattr(first, "language", None)


class _NativeHuggingFaceBackend:
    def __init__(self, settings: Qwen3ASRSettings) -> None:
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise ASRDependencyError(
                "native-hf backend requires a Transformers release with Qwen3-ASR support"
            ) from exc
        device, dtype = resolve_torch_placement(torch, settings)
        kwargs: dict[str, Any] = {"dtype": dtype, "device_map": device}
        if settings.revision is not None:
            kwargs["revision"] = settings.revision
        try:
            self.processor = AutoProcessor.from_pretrained(settings.checkpoint, revision=settings.revision)
            self.model = AutoModelForMultimodalLM.from_pretrained(settings.checkpoint, **kwargs)
        except Exception as exc:
            raise ASRInitializationError(f"failed to load {settings.checkpoint}: {exc}") from exc
        if not configure_greedy_generation(self.model):
            raise ASRInitializationError("native Transformers model has no generation_config")
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
        request: dict[str, Any] = {"audio": str(audio_path)}
        if language is not None:
            request["language"] = language
        inputs = self.processor.apply_transcription_request(**request)
        inputs = inputs.to(self.model.device, self.model.dtype)
        generation: dict[str, Any] = {
            "max_new_tokens": self.max_output_tokens,
            "do_sample": False,
            "num_beams": 1,
        }
        if deadline is not None:
            generation["max_time"] = deadline.remaining_seconds()
        output_ids = self.model.generate(**inputs, **generation)
        if deadline is not None:
            deadline.check("native Hugging Face ASR inference")
        prompt_length = inputs["input_ids"].shape[1]
        parsed = self.processor.decode(output_ids[:, prompt_length:], return_format="parsed")[0]
        if not isinstance(parsed, Mapping):
            raise TranscriptionError("native Transformers processor returned an unexpected result")
        return str(parsed.get("transcription", "")), optional_string(parsed.get("language"))


_resolve_torch_placement = resolve_torch_placement
