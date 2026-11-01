"""Qwen3-ASR transcription boundary with optional, content-addressed caching.

Heavy ASR dependencies are imported lazily so text-only Phase 02 remains usable.
Audio paths are passed directly to Qwen without resampling or waveform rewriting.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Protocol

from .compliance import SampleDeadline, SampleDeadlineExceeded, configure_greedy_generation

ASRBackendName = Literal["qwen-asr-transformers", "native-hf"]
ASRDevice = Literal["auto", "cpu", "cuda"] | str
ASRDtype = Literal["auto", "float32", "float16", "bfloat16"]


class ASRError(RuntimeError):
    """Base class for explicit Phase 03 transcription failures."""


class AudioPathError(ASRError):
    """An audio reference is missing, invalid, or outside its permitted root."""


class ASRDependencyError(ASRError):
    """The selected optional Qwen3-ASR dependency stack is unavailable."""


class ASRInitializationError(ASRError):
    """The selected Qwen3-ASR model could not be initialized."""


class TranscriptionError(ASRError):
    """Qwen3-ASR inference failed or returned a malformed result."""


class TranscriptionCacheError(ASRError):
    """A cache entry exists but is corrupt or inconsistent."""


@dataclass(frozen=True)
class Qwen3ASRSettings:
    checkpoint: str = "Qwen/Qwen3-ASR-1.7B"
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
        """Return every setting that can change decoded transcription text."""

        return {
            "checkpoint": self.checkpoint,
            "revision": self.revision,
            "backend": self.backend,
            "language": self.language,
            "max_output_tokens": self.max_output_tokens,
            "dtype": self.dtype,
            "preprocessing_version": self.preprocessing_version,
        }


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    inference_latency_seconds: float
    audio_sha256: str
    checkpoint: str
    backend: str
    cached: bool = False


class Transcriber(Protocol):
    settings: Qwen3ASRSettings

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        deadline: SampleDeadline | None = None,
    ) -> TranscriptionResult: ...


class _LoadedBackend(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        *,
        deadline: SampleDeadline | None = None,
    ) -> tuple[str, str | None]: ...


BackendFactory = Callable[[Qwen3ASRSettings], _LoadedBackend]


class Qwen3ASRTranscriber:
    """Lazy, reusable Qwen3-ASR transcriber for local official audio files."""

    def __init__(
        self,
        settings: Qwen3ASRSettings | None = None,
        *,
        allowed_audio_root: Path | str | None = None,
        backend_factory: BackendFactory | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.settings = settings or Qwen3ASRSettings()
        self.allowed_audio_root = (
            Path(allowed_audio_root).expanduser().resolve() if allowed_audio_root is not None else None
        )
        self._backend_factory = backend_factory or _load_backend
        self._clock = clock
        self._backend: _LoadedBackend | None = None
        self.initialization_latency_seconds: float | None = None

    def initialize(self, *, deadline: SampleDeadline | None = None) -> None:
        """Load the model once. Subsequent calls reuse the same loaded object."""

        if self._backend is not None:
            return
        if deadline is not None:
            deadline.check("ASR model initialization")
        started = self._clock()
        try:
            self._backend = self._backend_factory(self.settings)
        except (ASRDependencyError, ASRInitializationError):
            raise
        except Exception as exc:
            raise ASRInitializationError(f"failed to initialize {self.settings.checkpoint}: {exc}") from exc
        self.initialization_latency_seconds = self._clock() - started
        if deadline is not None:
            deadline.check("ASR model initialization")

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        deadline: SampleDeadline | None = None,
    ) -> TranscriptionResult:
        if deadline is not None:
            deadline.check("audio path validation")
        resolved = resolve_audio_path(audio_path, allowed_root=self.allowed_audio_root)
        fingerprint = _sha256_file(resolved)
        if deadline is not None:
            deadline.check("audio fingerprinting")
        self.initialize(deadline=deadline)
        assert self._backend is not None
        started = self._clock()
        try:
            text, language = self._backend.transcribe(
                resolved,
                self.settings.language,
                deadline=deadline,
            )
        except SampleDeadlineExceeded:
            raise
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"Qwen3-ASR inference failed for {resolved.name}: {exc}") from exc
        latency = self._clock() - started
        if deadline is not None:
            deadline.check("ASR inference")
        if not isinstance(text, str) or not text.strip():
            raise TranscriptionError(f"Qwen3-ASR returned an empty transcription for {resolved.name}")
        return TranscriptionResult(
            text=text.strip(),
            language=language,
            inference_latency_seconds=latency,
            audio_sha256=fingerprint,
            checkpoint=self.settings.checkpoint,
            backend=self.settings.backend,
        )


class CachedTranscriber:
    """Content-addressed JSON cache that stores successful transcriptions only."""

    def __init__(self, inner: Transcriber, cache_dir: Path | str) -> None:
        self.inner = inner
        self.settings = inner.settings
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.hits = 0
        self.misses = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        deadline: SampleDeadline | None = None,
    ) -> TranscriptionResult:
        if deadline is not None:
            deadline.check("transcription cache lookup")
        allowed_root = getattr(self.inner, "allowed_audio_root", None)
        resolved = resolve_audio_path(audio_path, allowed_root=allowed_root)
        audio_sha256 = _sha256_file(resolved)
        key = _cache_key(audio_sha256, self.settings.cache_identity())
        entry_path = self.cache_dir / key[:2] / f"{key}.json"
        if entry_path.exists():
            result = self._read_entry(entry_path, key, audio_sha256)
            self.hits += 1
            if deadline is not None:
                deadline.check("transcription cache lookup")
            return replace(result, cached=True, inference_latency_seconds=0.0)

        self.misses += 1
        result = self.inner.transcribe(resolved, deadline=deadline)
        if result.audio_sha256 != audio_sha256:
            raise TranscriptionCacheError("transcriber audio fingerprint changed during inference")
        self._write_entry(entry_path, key, result)
        if deadline is not None:
            deadline.check("transcription cache write")
        return result

    def _read_entry(self, path: Path, expected_key: str, audio_sha256: str) -> TranscriptionResult:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            result_raw = raw["result"]
            if raw["cache_key"] != expected_key or result_raw["audio_sha256"] != audio_sha256:
                raise KeyError("cache identity mismatch")
            result = TranscriptionResult(**result_raw)
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise TranscriptionCacheError(f"invalid transcription cache entry {path}: {exc}") from exc
        if not result.text.strip():
            raise TranscriptionCacheError(f"empty transcription cache entry {path}")
        return result

    def _write_entry(self, path: Path, key: str, result: TranscriptionResult) -> None:
        payload = {
            "schema_version": 1,
            "cache_key": key,
            "settings": dict(self.settings.cache_identity()),
            "result": asdict(result),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
                temporary = handle.name
            os.replace(temporary, path)
        except OSError as exc:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
            raise TranscriptionCacheError(f"could not write transcription cache {path}: {exc}") from exc


def resolve_audio_path(audio_path: Path | str, *, allowed_root: Path | str | None = None) -> Path:
    candidate = Path(audio_path).expanduser()
    root = Path(allowed_root).expanduser().resolve() if allowed_root is not None else None
    if not candidate.is_absolute() and root is not None:
        candidate = root / candidate
    resolved = candidate.resolve()
    if root is not None and not resolved.is_relative_to(root):
        raise AudioPathError(f"audio path escapes permitted split root: {audio_path}")
    if not resolved.is_file():
        raise AudioPathError(f"audio file not found: {resolved}")
    return resolved


def _load_backend(settings: Qwen3ASRSettings) -> _LoadedBackend:
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
                "qwen-asr backend is not installed; install the Phase 03 ASR dependency group"
            ) from exc
        device, dtype = _resolve_torch_placement(torch, settings)
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
        device, dtype = _resolve_torch_placement(torch, settings)
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
        return str(parsed.get("transcription", "")), _optional_string(parsed.get("language"))


def _resolve_torch_placement(torch: Any, settings: Qwen3ASRSettings) -> tuple[str, Any]:
    if settings.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    elif settings.device == "cuda":
        device = "cuda:0"
    else:
        device = str(settings.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ASRInitializationError("CUDA was requested but torch.cuda.is_available() is false")

    if settings.dtype == "auto":
        if device.startswith("cuda"):
            supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            dtype = torch.bfloat16 if supports_bf16 else torch.float16
        else:
            dtype = torch.float32
    else:
        dtype = getattr(torch, settings.dtype)
    if device == "cpu" and dtype == torch.float16:
        raise ASRInitializationError("float16 is not supported for the CPU ASR backend; use float32 or auto")
    return device, dtype


def _cache_key(audio_sha256: str, identity: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"audio_sha256": audio_sha256, "settings": identity},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
