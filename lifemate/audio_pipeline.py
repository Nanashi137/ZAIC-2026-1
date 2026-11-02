"""Phase 03 audio/history orchestration over the unchanged Phase 02 runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .asr import Transcriber, TranscriptionResult
from .compliance import SampleDeadline
from .data import DatasetReader, Sample
from .runtime import DecisionSource, RunResult, Runtime


@dataclass(frozen=True)
class AudioRunDiagnostics:
    transcribed_turns: int
    initialization_latency_seconds: float
    final_audio_latency_seconds: float
    history_audio_latency_seconds: float
    cache_hits: int
    cache_hit_rate: float
    failure_count: int


@dataclass(frozen=True)
class AudioRunResult:
    run: RunResult
    final_transcription: TranscriptionResult
    history_transcriptions: tuple[TranscriptionResult, ...]
    conversation_history: tuple[Mapping[str, Any], ...]
    diagnostics: AudioRunDiagnostics


class AudioInferencePipeline:
    """Transcribe one sample, reconstruct history, then invoke the existing runtime."""

    def __init__(self, dataset: DatasetReader, runtime: Runtime, transcriber: Transcriber) -> None:
        if runtime.dataset.dataset_root != dataset.dataset_root or runtime.dataset.split != dataset.split:
            raise ValueError("audio pipeline and runtime must use the same dataset split")
        self.dataset = dataset
        self.runtime = runtime
        self.transcriber = transcriber

    def run(
        self,
        sample_id: str,
        decisions: DecisionSource,
        *,
        deadline: SampleDeadline | None = None,
    ) -> AudioRunResult:
        if deadline is not None:
            deadline.check("audio sample setup")
        sample = self.dataset.get_sample(sample_id)
        initialization_before = _initialization_latency(self.transcriber)
        final = self.transcriber.transcribe(self._audio_path(sample.audio), deadline=deadline)
        initialization_after = _initialization_latency(self.transcriber)
        history, history_results = self._transcribe_history(sample, deadline=deadline)
        run = self.runtime.run(
            sample.id,
            final.text,
            decisions,
            conversation_history=history,
            deadline=deadline,
        )
        all_results = (final, *history_results)
        cache_hits = sum(result.cached for result in all_results)
        return AudioRunResult(
            run=run,
            final_transcription=final,
            history_transcriptions=history_results,
            conversation_history=history,
            diagnostics=AudioRunDiagnostics(
                transcribed_turns=len(all_results),
                initialization_latency_seconds=(
                    initialization_after or 0.0 if initialization_before is None else 0.0
                ),
                final_audio_latency_seconds=final.inference_latency_seconds,
                history_audio_latency_seconds=sum(
                    result.inference_latency_seconds for result in history_results if not result.cached
                ),
                cache_hits=cache_hits,
                cache_hit_rate=cache_hits / len(all_results),
                failure_count=0,
            ),
        )

    def _transcribe_history(
        self,
        sample: Sample,
        *,
        deadline: SampleDeadline | None = None,
    ) -> tuple[tuple[Mapping[str, Any], ...], tuple[TranscriptionResult, ...]]:
        reconstructed: list[Mapping[str, Any]] = []
        transcriptions: list[TranscriptionResult] = []
        for index, turn in enumerate(sample.history):
            role = turn.get("role")
            if role == "user":
                if deadline is not None:
                    deadline.check("history audio transcription")
                audio = turn.get("audio")
                if not isinstance(audio, str) or not audio:
                    raise ValueError(f"sample {sample.id!r} history turn {index} has no user audio reference")
                result = self.transcriber.transcribe(self._audio_path(audio), deadline=deadline)
                transcriptions.append(result)
                reconstructed.append(
                    MappingProxyType(
                        {
                            "role": "user",
                            "transcript": result.text,
                            "language": result.language,
                        }
                    )
                )
                continue
            if role == "assistant":
                reconstructed.append(
                    MappingProxyType(
                        {
                            "role": "assistant",
                            "text": turn.get("text"),
                            "tool_calls": _preserve(turn.get("tool_calls")),
                            "tool_results": _preserve(turn.get("tool_results")),
                        }
                    )
                )
                continue
            raise ValueError(f"sample {sample.id!r} history turn {index} has unknown role {role!r}")
        return tuple(reconstructed), tuple(transcriptions)

    def _audio_path(self, reference: str) -> Path:
        """Resolve only beneath the selected public split, never train or environment state."""

        split_root = self.dataset.split_dir.resolve()
        path = (split_root / reference).resolve()
        if not path.is_relative_to(split_root):
            raise ValueError(f"audio reference escapes split directory: {reference!r}")
        return path


def _preserve(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _preserve(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_preserve(item) for item in value)
    return value


def _initialization_latency(transcriber: Any) -> float | None:
    current = transcriber
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        latency = getattr(current, "initialization_latency_seconds", None)
        if isinstance(latency, (int, float)) and not isinstance(latency, bool) and latency >= 0:
            return float(latency)
        inner = getattr(current, "inner", None)
        if inner is None:
            break
        current = inner
    return None
