from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lifemate.asr import (
    ASRInitializationError,
    AudioPathError,
    CachedTranscriber,
    Qwen3ASRSettings,
    Qwen3ASRTranscriber,
    TranscriptionError,
)


class FakeBackend:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[Path, str | None]] = []

    def transcribe(self, audio_path: Path, language: str | None, *, deadline=None) -> tuple[str, str | None]:
        del deadline
        self.calls.append((audio_path, language))
        if self.fail:
            raise RuntimeError("decoder failed")
        return "Xin chào", language or "Vietnamese"


def _audio(tmp_path: Path, name: str = "sample.wav", content: bytes = b"RIFF-fixture") -> Path:
    path = tmp_path / name
    path.write_bytes(content)
    return path


def test_qwen_initializes_once_reuses_model_and_preserves_language(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    backend = FakeBackend()
    factory_calls: list[Qwen3ASRSettings] = []

    def factory(settings: Qwen3ASRSettings) -> FakeBackend:
        factory_calls.append(settings)
        return backend

    transcriber = Qwen3ASRTranscriber(
        Qwen3ASRSettings(language="Vietnamese"),
        allowed_audio_root=tmp_path,
        backend_factory=factory,
    )

    first = transcriber.transcribe(audio)
    second = transcriber.transcribe(audio)

    assert len(factory_calls) == 1
    assert len(backend.calls) == 2
    assert first.text == second.text == "Xin chào"
    assert first.language == "Vietnamese"
    assert first.audio_sha256 == second.audio_sha256
    assert transcriber.initialization_latency_seconds is not None


def test_default_checkpoint_stays_within_competition_model_budget() -> None:
    assert Qwen3ASRSettings().checkpoint == "Qwen/Qwen3-ASR-0.6B"


def test_audio_path_must_exist_and_stay_inside_allowed_root(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = _audio(tmp_path, "outside.wav")
    transcriber = Qwen3ASRTranscriber(allowed_audio_root=allowed, backend_factory=lambda _: FakeBackend())

    with pytest.raises(AudioPathError, match="escapes"):
        transcriber.transcribe(outside)
    with pytest.raises(AudioPathError, match="not found"):
        transcriber.transcribe(allowed / "missing.wav")


def test_initialization_and_inference_failures_are_explicit(tmp_path: Path) -> None:
    audio = _audio(tmp_path)

    def broken_factory(_settings: Qwen3ASRSettings) -> Any:
        raise OSError("load failed")

    broken = Qwen3ASRTranscriber(backend_factory=broken_factory)
    with pytest.raises(ASRInitializationError, match="load failed"):
        broken.transcribe(audio)

    failed = Qwen3ASRTranscriber(backend_factory=lambda _: FakeBackend(fail=True))
    with pytest.raises(TranscriptionError, match="decoder failed"):
        failed.transcribe(audio)


def test_backend_checkpoint_variants_cannot_be_mixed() -> None:
    with pytest.raises(ValueError, match="requires"):
        Qwen3ASRSettings(backend="native-hf", checkpoint="Qwen/Qwen3-ASR-1.7B")
    with pytest.raises(ValueError, match="cannot use"):
        Qwen3ASRSettings(backend="qwen-asr-transformers", checkpoint="Qwen/Qwen3-ASR-1.7B-hf")


def test_cache_reuses_success_by_audio_content_and_settings(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    backend = FakeBackend()
    inner = Qwen3ASRTranscriber(backend_factory=lambda _: backend)
    cache = CachedTranscriber(inner, tmp_path / "cache")

    first = cache.transcribe(audio)
    second = cache.transcribe(audio)

    assert not first.cached
    assert second.cached
    assert len(backend.calls) == 1
    assert cache.hits == cache.misses == 1
    assert cache.hit_rate == 0.5

    audio.write_bytes(b"changed-audio")
    changed = cache.transcribe(audio)
    assert not changed.cached
    assert len(backend.calls) == 2

    other_backend = FakeBackend()
    other_settings = Qwen3ASRSettings(language=None)
    other_cache = CachedTranscriber(
        Qwen3ASRTranscriber(other_settings, backend_factory=lambda _: other_backend),
        tmp_path / "cache",
    )
    other_cache.transcribe(audio)
    assert len(other_backend.calls) == 1
    assert other_cache.misses == 1


def test_cache_never_records_failed_transcription(tmp_path: Path) -> None:
    audio = _audio(tmp_path)
    backend = FakeBackend(fail=True)
    cache = CachedTranscriber(Qwen3ASRTranscriber(backend_factory=lambda _: backend), tmp_path / "cache")

    with pytest.raises(TranscriptionError):
        cache.transcribe(audio)
    with pytest.raises(TranscriptionError):
        cache.transcribe(audio)

    assert len(backend.calls) == 2
    assert cache.hits == 0
    assert cache.misses == 2


def test_requested_cuda_is_rejected_when_unavailable() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

    class FakeTorch:
        cuda = FakeCuda()
        float16 = "float16"
        float32 = "float32"
        bfloat16 = "bfloat16"

    from lifemate.asr import _resolve_torch_placement

    with pytest.raises(ASRInitializationError, match="CUDA was requested"):
        _resolve_torch_placement(FakeTorch(), Qwen3ASRSettings(device="cuda"))

    device, dtype = _resolve_torch_placement(FakeTorch(), Qwen3ASRSettings(device="cpu", dtype="auto"))
    assert (device, dtype) == ("cpu", "float32")
