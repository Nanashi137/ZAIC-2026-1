from __future__ import annotations

import pytest

from lifemate.cli import (
    _asr_run_identity,
    _build_base_asr_transcriber,
    build_batch_parser,
    build_evaluate_parser,
    build_parser,
    main,
)


def test_powershell_friendly_single_tool_arguments() -> None:
    args = build_parser().parse_args(
        [
            "--sample-id",
            "public_0039",
            "--transcript",
            "debug",
            "--tool",
            "alarm.list",
            "--arguments-json",
            "{}",
        ]
    )

    assert args.tool == "alarm.list"
    assert args.arguments_json == {}


def test_remote_model_endpoint_requires_explicit_data_approval() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--sample-id",
                "public_0039",
                "--transcript",
                "debug",
                "--mode",
                "model",
                "--model",
                "remote-model",
                "--base-url",
                "https://provider.example/v1",
            ]
        )


def test_batch_cli_contract_and_backward_compatible_aliases() -> None:
    batch = build_batch_parser().parse_args(
        ["--output", "outputs/submission.jsonl", "--limit", "5", "--timeout", "30"]
    )
    assert batch.model == "gpt-4.1-mini"
    assert batch.base_url == "https://api.openai.com/v1"
    assert batch.workers == 1
    assert batch.limit == 5
    assert batch.asr_checkpoint == "Qwen/Qwen3-ASR-0.6B"
    assert batch.asr_engine == "qwen"
    assert batch.agent_architecture == "direct"
    assert batch.prompt_version == "baseline"
    assert batch.invalid_response_history == "full"

    evaluation = build_evaluate_parser().parse_args(["--output-dir", "outputs/evaluation"])
    assert evaluation.asr_checkpoint == "Qwen/Qwen3-ASR-0.6B"
    assert evaluation.asr_engine == "qwen"

    experiment = build_evaluate_parser().parse_args([
        "--output-dir", "outputs/plan-evaluation",
        "--agent-architecture", "plan-ledger",
        "--prompt-version", "grounded-v2",
        "--invalid-response-history", "compact",
    ])
    assert experiment.agent_architecture == "plan-ledger"
    assert experiment.prompt_version == "grounded-v2"
    assert experiment.invalid_response_history == "compact"

    single = build_parser().parse_args(
        ["--sample-id", "public_0001", "--transcript", "debug", "--max-model-tokens", "321"]
    )
    assert single.max_output_tokens == 321


def test_evaluation_cli_builds_chunkformer_ctc_without_rnnt_settings(tmp_path) -> None:
    args = build_evaluate_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--input-mode",
            "asr",
            "--asr-engine",
            "chunkformer-ctc",
            "--asr-checkpoint",
            "khanhld/chunkformer-ctc-large-vie",
            "--asr-device",
            "cuda",
            "--asr-dtype",
            "float32",
        ]
    )

    settings, transcriber = _build_base_asr_transcriber(args, tmp_path)
    identity = _asr_run_identity(settings, args)

    assert transcriber.settings is settings
    assert settings.backend == "chunkformer-ctc"
    assert settings.rnnt_max_symbols_per_frame is None
    assert identity["engine"] == "chunkformer-ctc"
    assert identity["rnnt_max_symbols_per_frame"] is None
