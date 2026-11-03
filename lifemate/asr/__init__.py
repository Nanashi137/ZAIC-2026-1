"""Local ASR backends with a shared, cache-compatible transcriber contract."""

from .chunkformer import ChunkFormerSettings, ChunkFormerTranscriber
from .common import (
    ASRDependencyError,
    ASRError,
    ASRInitializationError,
    AudioPathError,
    CachedTranscriber,
    Transcriber,
    TranscriptionCacheError,
    TranscriptionError,
    TranscriptionResult,
    audio_duration_seconds,
    resolve_audio_path,
)
from .phowhisper import PhoWhisperSettings, PhoWhisperTranscriber
from .qwen import Qwen3ASRSettings, Qwen3ASRTranscriber, _resolve_torch_placement
from .whisper import (
    FasterWhisperSettings,
    FasterWhisperTranscriber,
    WhisperSettings,
    WhisperTranscriber,
)

__all__ = [
    "ASRDependencyError",
    "ASRError",
    "ASRInitializationError",
    "AudioPathError",
    "CachedTranscriber",
    "ChunkFormerSettings",
    "ChunkFormerTranscriber",
    "FasterWhisperSettings",
    "FasterWhisperTranscriber",
    "PhoWhisperSettings",
    "PhoWhisperTranscriber",
    "Qwen3ASRSettings",
    "Qwen3ASRTranscriber",
    "Transcriber",
    "TranscriptionCacheError",
    "TranscriptionError",
    "TranscriptionResult",
    "WhisperSettings",
    "WhisperTranscriber",
    "audio_duration_seconds",
    "resolve_audio_path",
    "_resolve_torch_placement",
]
