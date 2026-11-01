from __future__ import annotations

import pytest

from lifemate.cli import build_batch_parser, build_parser, main


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

    single = build_parser().parse_args(
        ["--sample-id", "public_0001", "--transcript", "debug", "--max-model-tokens", "321"]
    )
    assert single.max_output_tokens == 321
