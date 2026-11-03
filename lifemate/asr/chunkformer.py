"""ChunkFormer CTC and RNN-T adapters for Vietnamese ASR."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

from tqdm.auto import tqdm as Tqdm

from ..compliance import SampleDeadline
from .common import (
    ASRDependencyError,
    ASRDevice,
    ASRDtype,
    ASRInitializationError,
    BackendFactory,
    BaseASRTranscriber,
    TranscriptionError,
    model_parameter_count,
    resolve_torch_placement,
)

CHUNKFORMER_RNNT_BACKEND = "chunkformer-rnnt"
CHUNKFORMER_CTC_BACKEND = "chunkformer-ctc"


@dataclass(frozen=True)
class ChunkFormerSettings:
    checkpoint: str = "khanhld/chunkformer-rnnt-large-vie"
    revision: str | None = None
    backend: str = CHUNKFORMER_RNNT_BACKEND
    device: ASRDevice = "auto"
    dtype: ASRDtype = "float32"
    language: str | None = "vi"
    chunk_size: int = 64
    left_context_size: int = 128
    right_context_size: int = 128
    total_batch_duration: int = 1800
    rnnt_max_symbols_per_frame: int | None = 2
    preprocessing_version: str = "chunkformer-endless-decode-v2"

    def __post_init__(self) -> None:
        if not self.checkpoint.strip():
            raise ValueError("checkpoint must be non-empty")
        if self.backend not in {CHUNKFORMER_RNNT_BACKEND, CHUNKFORMER_CTC_BACKEND}:
            raise ValueError(f"unsupported ChunkFormer backend: {self.backend}")
        if self.revision is not None:
            raise ValueError("the installed ChunkFormer API does not expose a revision argument")
        if self.chunk_size < 1 or self.left_context_size < 0 or self.right_context_size < 0:
            raise ValueError("ChunkFormer chunk/context sizes are invalid")
        if self.total_batch_duration <= 0:
            raise ValueError("total_batch_duration must be positive")
        if self.backend == CHUNKFORMER_RNNT_BACKEND:
            if self.rnnt_max_symbols_per_frame is None or self.rnnt_max_symbols_per_frame < 1:
                raise ValueError("RNN-T requires a positive rnnt_max_symbols_per_frame")
        elif self.rnnt_max_symbols_per_frame is not None:
            raise ValueError("CTC does not use rnnt_max_symbols_per_frame; set it to None")

    def cache_identity(self) -> Mapping[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "backend": self.backend,
            "language": self.language,
            "chunk_size": self.chunk_size,
            "left_context_size": self.left_context_size,
            "right_context_size": self.right_context_size,
            "total_batch_duration": self.total_batch_duration,
            "rnnt_max_symbols_per_frame": self.rnnt_max_symbols_per_frame,
            "dtype": self.dtype,
            "preprocessing_version": self.preprocessing_version,
        }


class ChunkFormerTranscriber(BaseASRTranscriber):
    inference_name = "ChunkFormer"

    def __init__(
        self,
        settings: ChunkFormerSettings | None = None,
        *,
        allowed_audio_root: Path | str | None = None,
        backend_factory: BackendFactory | None = None,
        clock=None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "allowed_audio_root": allowed_audio_root,
            "backend_factory": backend_factory or _ChunkFormerBackend,
        }
        if clock is not None:
            kwargs["clock"] = clock
        super().__init__(settings or ChunkFormerSettings(), **kwargs)


class _ChunkFormerBackend:
    def __init__(self, settings: ChunkFormerSettings) -> None:
        try:
            import chunkformer.chunkformer_model as chunkformer_model_module
            import torch
            from chunkformer import ChunkFormerModel
        except ImportError as exc:
            raise ASRDependencyError(
                "ChunkFormer requires the chunkformer package; install the ASR benchmark dependencies"
            ) from exc
        device, dtype = resolve_torch_placement(torch, settings)
        try:
            with _quiet_chunkformer_progress(chunkformer_model_module):
                self.model = ChunkFormerModel.from_pretrained(settings.checkpoint)
            _validate_loaded_model_family(self.model, settings.backend)
            self.model.to(device=device, dtype=dtype)
            self.model.eval()
        except Exception as exc:
            raise ASRInitializationError(f"failed to load {settings.checkpoint}: {exc}") from exc
        self.settings = settings
        self._chunkformer_model_module = chunkformer_model_module
        self.parameter_count = model_parameter_count(self.model)
        self.parameter_count_source = "loaded_chunkformer_model.parameters()"

    def transcribe(
        self,
        audio_path: Path,
        language: str | None,
        *,
        deadline: SampleDeadline | None = None,
    ) -> tuple[str, str | None]:
        if deadline is not None:
            deadline.check("ChunkFormer inference")
        decode_kwargs = {
            "audio_path": str(audio_path),
            "chunk_size": self.settings.chunk_size,
            "left_context_size": self.settings.left_context_size,
            "right_context_size": self.settings.right_context_size,
            "total_batch_duration": self.settings.total_batch_duration,
            "return_timestamps": False,
        }
        with _quiet_chunkformer_progress(self._chunkformer_model_module):
            if self.settings.backend == CHUNKFORMER_RNNT_BACKEND:
                assert self.settings.rnnt_max_symbols_per_frame is not None
                with _bounded_rnnt_search(
                    self._chunkformer_model_module,
                    self.settings.rnnt_max_symbols_per_frame,
                ):
                    output = self.model.endless_decode(**decode_kwargs)
            else:
                output = self.model.endless_decode(**decode_kwargs)
        if deadline is not None:
            deadline.check("ChunkFormer inference")
        return _extract_text(output), language


def _validate_loaded_model_family(model: Any, backend: str) -> None:
    actual = getattr(getattr(model, "config", None), "model", None)
    expected = "transducer" if backend == CHUNKFORMER_RNNT_BACKEND else "asr_model"
    if actual != expected:
        raise ASRInitializationError(
            f"checkpoint model family mismatch: backend {backend!r} requires config.model={expected!r}, "
            f"got {actual!r}"
        )


_CHUNKFORMER_PATCH_LOCK = RLock()
_MISSING = object()


@contextmanager
def _quiet_chunkformer_progress(module: Any):
    """Suppress ChunkFormer/Hugging Face bars so callers can own one aggregate bar."""

    with _CHUNKFORMER_PATCH_LOCK:
        original_tqdm = getattr(module, "tqdm", _MISSING)
        original_snapshot_download = getattr(module, "snapshot_download", _MISSING)

        def no_progress(iterable: Any, *_args: Any, **_kwargs: Any) -> Any:
            return iterable

        class DisabledTqdm(Tqdm):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)

        module.tqdm = no_progress
        if callable(original_snapshot_download):
            def quiet_snapshot_download(*args: Any, **kwargs: Any) -> Any:
                kwargs["tqdm_class"] = DisabledTqdm
                return original_snapshot_download(*args, **kwargs)

            module.snapshot_download = quiet_snapshot_download
        try:
            yield
        finally:
            if original_tqdm is _MISSING:
                delattr(module, "tqdm")
            else:
                module.tqdm = original_tqdm
            if original_snapshot_download is not _MISSING:
                module.snapshot_download = original_snapshot_download


@contextmanager
def _bounded_rnnt_search(module: Any, max_symbols_per_frame: int):
    """Temporarily cap ChunkFormer 1.2.2's unexposed RNN-T greedy-search loop."""

    with _CHUNKFORMER_PATCH_LOCK:
        original_search = getattr(module, "optimized_search", None)
        if not callable(original_search):
            raise ASRDependencyError(
                "the installed ChunkFormer package does not expose optimized_search; "
                "the pinned decoder-cap integration must be reviewed"
            )

        def bounded_search(
            model: Any,
            encoder_out: Any,
            encoder_out_lens: Any,
            n_steps: int = max_symbols_per_frame,
        ) -> Any:
            return original_search(
                model,
                encoder_out,
                encoder_out_lens,
                n_steps=min(n_steps, max_symbols_per_frame),
            )

        module.optimized_search = bounded_search
        try:
            yield
        finally:
            module.optimized_search = original_search


def _extract_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, Mapping):
        for key in ("text", "transcription", "transcript"):
            value = output.get(key)
            if isinstance(value, str):
                return value
    if isinstance(output, Sequence) and not isinstance(output, (bytes, bytearray)):
        parts: list[str] = []
        for item in output:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                value = next(
                    (
                        item.get(key)
                        for key in ("text", "transcription", "transcript")
                        if isinstance(item.get(key), str)
                    ),
                    None,
                )
                if value is not None:
                    parts.append(value)
        if parts:
            return " ".join(part.strip() for part in parts if part.strip())
    raise TranscriptionError("ChunkFormer returned an unexpected transcription result")
