"""Shared ASR contracts, lazy initialization, path safety, and caching."""

from __future__ import annotations

import hashlib
import json
import os
import time
import wave
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Protocol

from ..compliance import SampleDeadline, SampleDeadlineExceeded

ASRDevice = str
ASRDtype = str

_EXACT_WHISPER_PARAMETER_COUNTS = {
    # Reproducibly derived from the official Transformers configs after tie_weights().
    "openai/whisper-medium": 763_857_920,
    "openai/whisper-large-v3": 1_543_490_560,
}


class ASRError(RuntimeError):
    """Base class for explicit transcription failures."""


class AudioPathError(ASRError):
    """An audio reference is missing, invalid, or outside its permitted root."""


class ASRDependencyError(ASRError):
    """The selected optional ASR dependency stack is unavailable."""


class ASRInitializationError(ASRError):
    """The selected ASR model could not be initialized."""


class TranscriptionError(ASRError):
    """ASR inference failed or returned a malformed result."""


class TranscriptionCacheError(ASRError):
    """A cache entry exists but is corrupt or inconsistent."""


class ASRSettings(Protocol):
    checkpoint: str
    backend: str
    device: ASRDevice
    dtype: ASRDtype

    def cache_identity(self) -> Mapping[str, Any]: ...


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
    settings: ASRSettings

    def transcribe(
        self,
        audio_path: Path | str,
        *,
        deadline: SampleDeadline | None = None,
    ) -> TranscriptionResult: ...


class LoadedBackend(Protocol):
    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        *,
        deadline: SampleDeadline | None = None,
    ) -> tuple[str, str | None]: ...


BackendFactory = Callable[[Any], LoadedBackend]


class BaseASRTranscriber:
    """Lazy, reusable transcriber shared by all local ASR adapters."""

    inference_name = "ASR"

    def __init__(
        self,
        settings: ASRSettings,
        *,
        allowed_audio_root: Path | str | None = None,
        backend_factory: BackendFactory,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.settings = settings
        self.allowed_audio_root = (
            Path(allowed_audio_root).expanduser().resolve() if allowed_audio_root is not None else None
        )
        self._backend_factory = backend_factory
        self._clock = clock
        self._backend: LoadedBackend | None = None
        self.initialization_latency_seconds: float | None = None

    @property
    def parameter_count(self) -> int | None:
        value = getattr(self._backend, "parameter_count", None)
        return value if isinstance(value, int) and value >= 0 else None

    @property
    def parameter_count_source(self) -> str | None:
        value = getattr(self._backend, "parameter_count_source", None)
        return value if isinstance(value, str) and value else None

    def initialize(self, *, deadline: SampleDeadline | None = None) -> None:
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
        fingerprint = sha256_file(resolved)
        if deadline is not None:
            deadline.check("audio fingerprinting")
        self.initialize(deadline=deadline)
        assert self._backend is not None
        started = self._clock()
        try:
            text, language = self._backend.transcribe(
                resolved,
                getattr(self.settings, "language", None),
                deadline=deadline,
            )
        except SampleDeadlineExceeded:
            raise
        except TranscriptionError:
            raise
        except Exception as exc:
            raise TranscriptionError(f"{self.inference_name} inference failed for {resolved.name}: {exc}") from exc
        latency = self._clock() - started
        if deadline is not None:
            deadline.check("ASR inference")
        if not isinstance(text, str) or not text.strip():
            raise TranscriptionError(f"{self.inference_name} returned an empty transcription for {resolved.name}")
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
        audio_sha256 = sha256_file(resolved)
        key = cache_key(audio_sha256, self.settings.cache_identity())
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


def audio_duration_seconds(path: Path | str) -> float:
    """Read PCM WAV duration without decoding or changing the audio."""

    resolved = Path(path)
    try:
        with wave.open(str(resolved), "rb") as handle:
            frame_rate = handle.getframerate()
            if frame_rate <= 0:
                raise AudioPathError(f"invalid WAV frame rate: {resolved}")
            return handle.getnframes() / frame_rate
    except (OSError, EOFError, wave.Error) as exc:
        raise AudioPathError(f"could not read WAV duration for {resolved}: {exc}") from exc


def resolve_torch_placement(torch: Any, settings: ASRSettings) -> tuple[str, Any]:
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
        try:
            dtype = getattr(torch, settings.dtype)
        except AttributeError as exc:
            raise ASRInitializationError(f"unsupported torch dtype: {settings.dtype}") from exc
    if device == "cpu" and dtype == torch.float16:
        raise ASRInitializationError("float16 is not supported for the CPU ASR backend; use float32 or auto")
    return device, dtype


def model_parameter_count(model: Any) -> int | None:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return None
    try:
        return sum(int(parameter.numel()) for parameter in parameters())
    except (AttributeError, TypeError, RuntimeError):
        return None


def transformers_architecture_parameter_count(checkpoint: str, revision: str | None = None) -> int:
    """Count logical parameters from a Transformers config without loading weight tensors."""

    if revision is None and checkpoint.lower() in _EXACT_WHISPER_PARAMETER_COUNTS:
        return _EXACT_WHISPER_PARAMETER_COUNTS[checkpoint.lower()]

    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForSpeechSeq2Seq
    except ImportError as exc:
        raise ASRDependencyError(
            "exact faster-whisper parameter counting requires accelerate and transformers"
        ) from exc
    try:
        config = AutoConfig.from_pretrained(checkpoint, revision=revision)
        with init_empty_weights():
            model = AutoModelForSpeechSeq2Seq.from_config(config)
        model.tie_weights()
        count = model_parameter_count(model)
    except Exception as exc:
        raise ASRInitializationError(
            f"could not derive exact parameter count from {checkpoint}: {exc}"
        ) from exc
    if count is None:
        raise ASRInitializationError(f"could not derive exact parameter count from {checkpoint}")
    return count


def cache_key(audio_sha256: str, identity: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"audio_sha256": audio_sha256, "settings": identity},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
