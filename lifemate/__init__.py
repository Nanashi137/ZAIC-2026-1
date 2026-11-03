"""Competition-compliant LifeMate text and Qwen3-ASR runtime."""

from .agent import ContextBuilder, LlmDecisionSource, ModelToolRegistry
from .asr import (
    CachedTranscriber,
    ChunkFormerSettings,
    ChunkFormerTranscriber,
    FasterWhisperSettings,
    FasterWhisperTranscriber,
    PhoWhisperSettings,
    PhoWhisperTranscriber,
    Qwen3ASRSettings,
    Qwen3ASRTranscriber,
    TranscriptionResult,
    WhisperSettings,
    WhisperTranscriber,
)
from .audio_pipeline import AudioInferencePipeline, AudioRunResult
from .batch import BatchConfig, BatchRunner, BatchRunSummary
from .compliance import (
    ReproducibilityReport,
    SampleDeadline,
    SampleDeadlineExceeded,
    configure_reproducibility,
)
from .data import Catalog, DatasetReader, Sample
from .model import InferenceSettings, OpenAICompatibleClient, RetrySettings, ScriptedModelClient
from .runtime import ExplicitActionSource, Runtime, RuntimeConfig, ToolAction
from .tool_schema import ToolSchemaCompatibilityError, adapt_function_parameters

__all__ = [
    "Catalog",
    "CachedTranscriber",
    "ChunkFormerSettings",
    "ChunkFormerTranscriber",
    "FasterWhisperSettings",
    "FasterWhisperTranscriber",
    "ContextBuilder",
    "DatasetReader",
    "ExplicitActionSource",
    "InferenceSettings",
    "LlmDecisionSource",
    "ModelToolRegistry",
    "OpenAICompatibleClient",
    "PhoWhisperSettings",
    "PhoWhisperTranscriber",
    "AudioInferencePipeline",
    "AudioRunResult",
    "BatchConfig",
    "BatchRunner",
    "BatchRunSummary",
    "Qwen3ASRSettings",
    "Qwen3ASRTranscriber",
    "ReproducibilityReport",
    "Runtime",
    "RuntimeConfig",
    "RetrySettings",
    "Sample",
    "SampleDeadline",
    "SampleDeadlineExceeded",
    "ScriptedModelClient",
    "ToolAction",
    "ToolSchemaCompatibilityError",
    "TranscriptionResult",
    "WhisperSettings",
    "WhisperTranscriber",
    "adapt_function_parameters",
    "configure_reproducibility",
]
