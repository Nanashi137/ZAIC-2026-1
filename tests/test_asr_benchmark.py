from __future__ import annotations

import json
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from lifemate.asr import (
    ASRInitializationError,
    ChunkFormerSettings,
    FasterWhisperSettings,
    PhoWhisperSettings,
    PhoWhisperTranscriber,
    TranscriptionError,
    TranscriptionResult,
)
from lifemate.asr.benchmark import _build_transcriber, build_parser, normalize_text, run_benchmark
from lifemate.asr.chunkformer import (
    CHUNKFORMER_CTC_BACKEND,
    CHUNKFORMER_RNNT_BACKEND,
    _bounded_rnnt_search,
    _ChunkFormerBackend,
    _extract_text,
    _quiet_chunkformer_progress,
    _validate_loaded_model_family,
)
from lifemate.asr.common import transformers_architecture_parameter_count
from lifemate.asr.whisper import FasterWhisperBackend, _faster_device


def _wav(path: Path, *, seconds: float = 0.1, rate: int = 8_000) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\0\0" * int(rate * seconds))
    return path


def test_phowhisper_rejects_over_window_before_loading_model(tmp_path: Path) -> None:
    audio = _wav(tmp_path / "long.wav", seconds=1.0)
    factory_calls = []
    transcriber = PhoWhisperTranscriber(
        PhoWhisperSettings(max_audio_seconds=0.5),
        backend_factory=lambda settings: factory_calls.append(settings),
    )

    with pytest.raises(TranscriptionError, match="no-chunking limit"):
        transcriber.transcribe(audio)

    assert factory_calls == []


def test_chunkformer_text_extraction_accepts_documented_and_segment_shapes() -> None:
    assert _extract_text("xin chào") == "xin chào"
    assert _extract_text({"text": "xin chào"}) == "xin chào"
    assert _extract_text([{"text": "xin"}, {"transcription": "chào"}]) == "xin chào"
    with pytest.raises(TranscriptionError, match="unexpected"):
        _extract_text({"tokens": [1, 2]})


def test_chunkformer_rnnt_cap_is_validated_and_part_of_cache_identity() -> None:
    settings = ChunkFormerSettings(rnnt_max_symbols_per_frame=2)
    assert settings.cache_identity()["rnnt_max_symbols_per_frame"] == 2
    assert settings.preprocessing_version == "chunkformer-endless-decode-v2"
    with pytest.raises(ValueError, match="rnnt_max_symbols_per_frame"):
        ChunkFormerSettings(rnnt_max_symbols_per_frame=0)


def test_chunkformer_ctc_has_distinct_identity_and_no_rnnt_cap() -> None:
    settings = ChunkFormerSettings(
        checkpoint="khanhld/chunkformer-ctc-large-vie",
        backend=CHUNKFORMER_CTC_BACKEND,
        rnnt_max_symbols_per_frame=None,
    )

    assert settings.cache_identity()["backend"] == CHUNKFORMER_CTC_BACKEND
    assert settings.cache_identity()["rnnt_max_symbols_per_frame"] is None
    with pytest.raises(ValueError, match="CTC does not use"):
        ChunkFormerSettings(backend=CHUNKFORMER_CTC_BACKEND)


def test_chunkformer_rejects_checkpoint_family_mismatch() -> None:
    _validate_loaded_model_family(
        SimpleNamespace(config=SimpleNamespace(model="transducer")),
        CHUNKFORMER_RNNT_BACKEND,
    )
    _validate_loaded_model_family(
        SimpleNamespace(config=SimpleNamespace(model="asr_model")),
        CHUNKFORMER_CTC_BACKEND,
    )
    with pytest.raises(ASRInitializationError, match="model family mismatch"):
        _validate_loaded_model_family(
            SimpleNamespace(config=SimpleNamespace(model="transducer")),
            CHUNKFORMER_CTC_BACKEND,
        )


def test_chunkformer_ctc_transcription_does_not_patch_rnnt_search(tmp_path: Path) -> None:
    module = SimpleNamespace(optimized_search=lambda *_args, **_kwargs: pytest.fail("RNN-T search was used"))

    class FakeModel:
        def endless_decode(self, **kwargs):
            assert kwargs["return_timestamps"] is False
            return "xin chào"

    backend = object.__new__(_ChunkFormerBackend)
    backend.settings = ChunkFormerSettings(
        checkpoint="test/ctc",
        backend=CHUNKFORMER_CTC_BACKEND,
        rnnt_max_symbols_per_frame=None,
    )
    backend.model = FakeModel()
    backend._chunkformer_model_module = module

    assert backend.transcribe(tmp_path / "sample.wav", "vi") == ("xin chào", "vi")


def test_benchmark_builds_explicit_chunkformer_ctc_settings(tmp_path: Path, monkeypatch) -> None:
    captured = []

    def fake_transcriber(settings, *, allowed_audio_root):
        captured.append((settings, allowed_audio_root))
        return SimpleNamespace(settings=settings)

    monkeypatch.setattr("lifemate.asr.benchmark.ChunkFormerTranscriber", fake_transcriber)
    args = build_parser().parse_args(
        [
            "--backend",
            CHUNKFORMER_CTC_BACKEND,
            "--checkpoint",
            "khanhld/chunkformer-ctc-large-vie",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    result = _build_transcriber(args, tmp_path)

    assert result.settings.backend == CHUNKFORMER_CTC_BACKEND
    assert result.settings.rnnt_max_symbols_per_frame is None
    assert captured[0][1] == tmp_path


def test_chunkformer_rnnt_cap_bounds_and_restores_internal_search() -> None:
    calls = []

    def original_search(model, encoder_out, encoder_out_lens, n_steps=64):
        calls.append((model, encoder_out, encoder_out_lens, n_steps))
        return n_steps

    module = SimpleNamespace(optimized_search=original_search)
    with _bounded_rnnt_search(module, 2):
        assert module.optimized_search("model", "encoded", "lengths") == 2
        assert module.optimized_search("model", "encoded", "lengths", n_steps=1) == 1
        assert module.optimized_search("model", "encoded", "lengths", n_steps=8) == 2

    assert module.optimized_search is original_search
    assert [call[-1] for call in calls] == [2, 1, 2]


def test_chunkformer_nested_progress_is_suppressed_and_restored() -> None:
    calls = []
    snapshot_calls = []

    def original_tqdm(iterable, *args, **kwargs):
        calls.append((args, kwargs))
        return iterable

    def original_snapshot_download(*args, **kwargs):
        snapshot_calls.append((args, kwargs))
        return "checkpoint"

    module = SimpleNamespace(tqdm=original_tqdm, snapshot_download=original_snapshot_download)
    with _quiet_chunkformer_progress(module):
        assert list(module.tqdm([1])) == [1]
        assert module.snapshot_download("repo") == "checkpoint"
        assert calls == []
        assert snapshot_calls[0][0] == ("repo",)
        assert snapshot_calls[0][1]["tqdm_class"]([1]).disable

    assert module.tqdm is original_tqdm
    assert module.snapshot_download is original_snapshot_download
    assert list(module.tqdm([2], desc="visible")) == [2]
    assert calls == [((), {"desc": "visible"})]


def test_shared_normalization_preserves_vietnamese_diacritics() -> None:
    assert normalize_text("  XIN, Chào—Bạn!  ") == "xin chào bạn"
    assert normalize_text("Hẹn lúc 09:30") == "hẹn lúc 09 30"


def test_faster_whisper_settings_require_an_exact_parameter_source() -> None:
    assert FasterWhisperSettings(checkpoint="medium").canonical_parameter_checkpoint == "openai/whisper-medium"
    assert FasterWhisperSettings(checkpoint="large-v3").canonical_parameter_checkpoint == "openai/whisper-large-v3"
    with pytest.raises(ValueError, match="parameter_count_checkpoint"):
        FasterWhisperSettings(checkpoint="custom/model")
    with pytest.raises(ValueError, match="supports"):
        FasterWhisperSettings(checkpoint="medium", dtype="bfloat16")
    assert _faster_device("cuda:2") == ("cuda", 2)
    assert transformers_architecture_parameter_count("openai/whisper-medium") == 763_857_920
    assert transformers_architecture_parameter_count("openai/whisper-large-v3") == 1_543_490_560


def test_faster_whisper_backend_consumes_generator_with_greedy_decoding(monkeypatch, tmp_path: Path) -> None:
    calls = []

    class FakeWhisperModel:
        def __init__(self, checkpoint, **kwargs) -> None:
            calls.append(("init", checkpoint, kwargs))

        def transcribe(self, audio, **kwargs):
            calls.append(("transcribe", audio, kwargs))
            return iter([SimpleNamespace(text=" xin "), SimpleNamespace(text="chào")]), SimpleNamespace(language="vi")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
    backend = FasterWhisperBackend(FasterWhisperSettings(checkpoint="medium", device="cpu", dtype="float32"))
    text, language = backend.transcribe(_wav(tmp_path / "sample.wav"), "vi")

    assert (text, language) == ("xin chào", "vi")
    assert backend.parameter_count == 763_857_920
    assert calls[1][2]["beam_size"] == 1
    assert calls[1][2]["best_of"] == 1
    assert calls[1][2]["vad_filter"] is False


def test_benchmark_checkpoints_each_success_and_resumes_only_pending(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "data"
    split_dir = dataset_root / "train"
    first_audio = _wav(split_dir / "audio" / "one.wav")
    _wav(split_dir / "audio" / "two.wav")
    rows = [
        {"id": "train_0001", "audio": "audio/one.wav", "audio_text": "xin chào"},
        {"id": "train_0002", "audio": "audio/two.wav", "audio_text": "mở nhạc"},
    ]
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output_dir = tmp_path / "benchmark"
    args = build_parser().parse_args(
        [
            "--dataset-root",
            str(dataset_root),
            "--backend",
            "qwen",
            "--checkpoint",
            "test/checkpoint",
            "--output-dir",
            str(output_dir),
            "--limit",
            "2",
            "--sample-id",
            "train_0001",
            "--sample-id",
            "train_0002",
            "--warmup-samples",
            "0",
            "--device",
            "cpu",
            "--dtype",
            "float32",
        ]
    )

    class FakeTranscriber:
        initialization_latency_seconds = 0.25
        parameter_count = 123
        parameter_count_source = "fake.parameters()"

        def __init__(self, fail_name: str | None) -> None:
            self.fail_name = fail_name
            self.calls: list[str] = []

        def initialize(self) -> None:
            return None

        def transcribe(self, audio_path: Path) -> TranscriptionResult:
            self.calls.append(audio_path.name)
            if audio_path.name == self.fail_name:
                raise RuntimeError("temporary decoder failure")
            text = "xin chào" if audio_path == first_audio else "mở nhạc"
            return TranscriptionResult(
                text=text,
                language="vi",
                inference_latency_seconds=0.01,
                audio_sha256="unused-by-benchmark",
                checkpoint="test/checkpoint",
                backend="fake",
            )

    first = FakeTranscriber("two.wav")
    monkeypatch.setattr("lifemate.asr.benchmark._build_transcriber", lambda *_: first)
    partial = run_benchmark(args)
    assert not partial["complete"]
    assert first.calls == ["one.wav", "two.wav"]
    assert len((output_dir / "checkpoint.jsonl").read_text(encoding="utf-8").splitlines()) == 1

    second = FakeTranscriber(None)
    monkeypatch.setattr("lifemate.asr.benchmark._build_transcriber", lambda *_: second)
    complete = run_benchmark(args)
    assert complete["complete"]
    assert second.calls == ["two.wav"]
    assert len((output_dir / "checkpoint.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["normalized"]["word_error_rate"] == 0.0
    assert metrics["normalized"]["char_error_rate"] == 0.0
    assert metrics["parameter_count"] == 123
    assert metrics["parameter_count_source"] == "fake.parameters()"
    assert "Exact parameter count: 123" in (output_dir / "report.md").read_text(encoding="utf-8")

    monkeypatch.setattr(
        "lifemate.asr.benchmark._build_transcriber",
        lambda *_: pytest.fail("a fully checkpointed resume must not load the model"),
    )
    run_benchmark(args)
    resumed_metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert resumed_metrics["initialization_latency_seconds"] == 0.25
    assert resumed_metrics["parameter_count"] == 123
