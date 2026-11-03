from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from lifemate.agent import LlmDecisionSource
from lifemate.cli import evaluate_main
from lifemate.data import DatasetReader
from lifemate.evaluation import (
    EvaluationSummary,
    _aggregate,
    _compare_calls,
    _write_jsonl,
    compare_evaluations,
    load_label_references,
    run_semantic_evaluation,
    select_sample_ids,
    write_comparison_markdown,
)
from lifemate.model import InferenceSettings, ModelResponse, ModelToolCall, ScriptedModelClient
from lifemate.runtime import ExplicitActionSource, Runtime, ToolAction
from lifemate.scoring import aggregate_scores, replay_gold_state_hash, score_sample
from tests.support import FakeFactory, FakeSession


def _train_dataset(tmp_path: Path) -> DatasetReader:
    split = tmp_path / "train"
    split.mkdir(parents=True)

    sample = {
        "id": "train_0001",
        "audio": "audio/train_0001/train_0001.wav",
        "audio_text": "list alarms",
        "context": {
            "installed_apps": {
                "alarm": {"tools": ["alarm.list"]},
            }
        },
        "history": [],
        "label": {
            "tool_calls": [{"name": "alarm.list", "arguments": {}}],
            "tool_results": [{"ok": True, "alarms": []}],
            "final_state": {},
        },
    }
    (split / "train.jsonl").write_text(json.dumps(sample) + "\n", encoding="utf-8")

    catalog = {
        "schema_version": "test",
        "apps": [
            {
                "id": "alarm",
                "tools": [
                    {
                        "name": "alarm.list",
                        "kind": "read",
                        "parameters": {"type": "object", "properties": {}},
                    },
                    {
                        "name": "sys.ask_missing_info",
                        "kind": "meta",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "tool_name": {"type": "string"},
                                "params": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["tool_name", "params"],
                        },
                    },
                ],
            }
        ],
    }
    (split / "train_tools_catalog.json").write_text(json.dumps(catalog), encoding="utf-8")
    return DatasetReader(tmp_path, "train", allow_train=True)


def test_semantic_evaluation_generates_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("lifemate.evaluation.replay_gold_state_hash", lambda *_, **__: "hash")
    progress_calls = []

    def fake_tqdm(iterable, **kwargs):
        progress_calls.append((iterable, kwargs))
        return iterable

    monkeypatch.setattr("lifemate.evaluation.tqdm", fake_tqdm)
    dataset = _train_dataset(tmp_path)
    references = load_label_references(dataset)
    selected = select_sample_ids(references, sample_ids=None, limit=1, seed=42)
    session = FakeSession({"alarm.list", "sys.ask_missing_info"}, state_hash="hash")
    runtime = Runtime(dataset, session_factory=FakeFactory(session))

    summary = run_semantic_evaluation(
        dataset=dataset,
        references=references,
        selected_ids=selected,
        runtime=runtime,
        decision_source_factory=lambda _sample: ExplicitActionSource([ToolAction("alarm.list", {})]),
        output_dir=tmp_path / "eval-out",
        input_mode="oracle",
        sample_timeout_seconds=10.0,
        compute_reference_state_hash=False,
        run_identity={"model": "fixture-model", "agent_output_mode": "native"},
    )

    assert summary.records_path.is_file()
    assert summary.metrics_path.is_file()
    assert summary.report_path.is_file()
    metrics = json.loads(summary.metrics_path.read_text(encoding="utf-8"))
    assert metrics["evaluated_samples"] == 1
    assert metrics["successful_samples"] == 1
    assert metrics["tool_sequence_exact_match_rate"] == 1.0
    assert metrics["challenge_score_exact"] is False
    assert "challenge_score" not in metrics
    assert metrics["run_identity"]["split"] == "train"
    assert metrics["run_identity"]["catalog_schema_version"] == "test"
    assert metrics["run_identity"]["selected_sample_count"] == 1
    assert len(metrics["run_identity"]["selected_ids_sha256"]) == 64
    run_config = json.loads((summary.records_path.parent / "run-config.json").read_text())
    assert run_config == metrics["run_identity"]
    assert progress_calls == [(
        selected,
        {
            "desc": "fixture-model native/oracle",
            "unit": "sample",
            "dynamic_ncols": True,
        },
    )]


def test_evaluate_cli_logs_root_config_before_inference(tmp_path: Path, monkeypatch) -> None:
    _train_dataset(tmp_path)
    output_dir = tmp_path / "evaluation"

    def fake_run(**kwargs):
        mode_dir = kwargs["output_dir"]
        return EvaluationSummary(
            mode_dir / "evaluation-records.jsonl",
            mode_dir / "evaluation-metrics.json",
            mode_dir / "evaluation-report.md",
            tuple(kwargs["selected_ids"]),
            {},
        )

    monkeypatch.setattr("lifemate.cli.configure_reproducibility", lambda *_, **__: None)
    monkeypatch.setattr("lifemate.cli.run_semantic_evaluation", fake_run)

    assert evaluate_main([
        "--dataset-root", str(tmp_path),
        "--output-dir", str(output_dir),
        "--base-url", "http://127.0.0.1:1990/v1",
        "--model", "fixture-model",
        "--limit", "1",
    ]) == 0

    config = json.loads((output_dir / "evaluation-config.json").read_text())
    assert config["model"] == "fixture-model"
    assert config["input_mode"] == "oracle"
    assert config["requested_modes"] == ["native"]
    assert config["selected_sample_count"] == 1
    assert config["asr"] is None
    assert "api_key" not in config


def test_sample_selection_rejects_negative_limit(tmp_path: Path) -> None:
    references = load_label_references(_train_dataset(tmp_path))

    with pytest.raises(ValueError, match="non-negative"):
        select_sample_ids(references, sample_ids=None, limit=-1, seed=42)


@pytest.mark.parametrize("output_mode", ["native", "constrained-json"])
@pytest.mark.parametrize("capture_raw_responses", [False, True])
def test_semantic_evaluation_serializes_model_request_diagnostics(
    tmp_path: Path, output_mode, capture_raw_responses: bool,
) -> None:
    dataset = _train_dataset(tmp_path)
    actions = [
        {"name": "alarm.list", "arguments": {}},
        {"name": "agent.finish", "arguments": {"status": "done", "reason": "complete"}},
    ]
    responses = [ModelResponse(content="not valid JSON")]
    for index, action in enumerate(actions):
        if output_mode == "native":
            responses.append(ModelResponse(tool_calls=(ModelToolCall(
                str(index), action["name"].replace(".", "__"), json.dumps(action["arguments"]),
            ),)))
        else:
            responses.append(ModelResponse(content=json.dumps(action)))
    source = LlmDecisionSource(
        ScriptedModelClient(responses), InferenceSettings(model="fixture-model", agent_output_mode=output_mode),
        capture_raw_responses=capture_raw_responses,
    )
    inference = FakeFactory(FakeSession({"alarm.list", "sys.ask_missing_info"}, state_hash="full"))
    reference = FakeFactory(FakeSession({"alarm.list"}, state_hash="full"))

    summary = run_semantic_evaluation(
        dataset=dataset, references=load_label_references(dataset), selected_ids=("train_0001",),
        runtime=Runtime(dataset, session_factory=inference), decision_source_factory=lambda _: source,
        output_dir=tmp_path / "model-eval", input_mode="oracle", sample_timeout_seconds=10,
        reference_session_factory=reference,
    )

    row = json.loads(summary.records_path.read_text(encoding="utf-8"))
    diagnostics = row["model_request_diagnostics"]
    assert diagnostics == json.loads(json.dumps(source.request_diagnostics))
    assert len(diagnostics) == row["model_requests"] == 3
    assert diagnostics[0]["validation_failure"] == (
        "missing_tool_call" if output_mode == "native" else "invalid_json_content"
    )
    assert ("raw_model_response" in diagnostics[0]) is capture_raw_responses
    if capture_raw_responses:
        assert diagnostics[0]["raw_model_response"] == {
            "content": "not valid JSON", "refusal": None, "tool_calls": [],
        }
    assert diagnostics[1]["selected_tool"] == "alarm.list"
    assert diagnostics[2]["selected_tool"] == "agent.finish"
    assert row["predicted_tool_calls"] == inference.session.get_tool_calls() == actions[:1]
    assert row["predicted_state_hash"] == "full"
    assert row["challenge_score_estimate"] == 1
    assert json.loads(summary.metrics_path.read_text())["challenge_score_estimate"] == 100
    assert "Challenge score estimate" in summary.report_path.read_text()
    assert len(inference.creations) == len(reference.creations) == 1


A = {"name": "alarm.list", "arguments": {}}
B = {"name": "alarm.create", "arguments": {"time": "07:00"}}
C = {"name": "alarm.create", "arguments": {"time": "08:00"}}


@pytest.mark.parametrize("gold,submitted,system_hash,expected", [
    ([], [], "different", 1.0),
    ([], [A], "hash", 0.0),
    ([A], [A], "hash", 1.0),
    ([A], [A], "different", 0.0),
    ([A], [B], "hash", 0.0),
    ([A, B], [A, B, C], "hash", 0.8),
    ([A], [A, A], "hash", 2 / 3),
    ([], None, None, 0.0),
    ([A], None, None, 0.0),
    ([A], [A], None, 0.0),
])
def test_challenge_formula(gold, submitted, system_hash, expected) -> None:
    result = score_sample(gold, submitted, system_hash, "hash")
    assert result["challenge_score_estimate"] == pytest.approx(expected)
    assert result["challenge_score_exact"] is False
    assert "challenge_score" not in result
    assert 0 <= result["challenge_score_estimate"] <= 1


def test_partial_precision_recall_and_duplicate_one_to_one() -> None:
    result = score_sample([A, B], [A, B, C], "hash", "hash")
    assert result["matched_calls"] == 2
    assert result["unmatched_calls"] == 1
    assert result["precision"] == pytest.approx(2 / 3)
    assert result["recall"] == 1.0
    duplicate = score_sample([A, A], [A, A, A], "hash", "hash")
    assert duplicate["matched_calls"] == 2
    assert duplicate["unmatched_calls"] == 1


def test_rejected_submitted_call_never_matches() -> None:
    result = score_sample([A, B], [A, B, C], "hash", "hash", rejected_indices=[1])
    assert result["matched_calls"] == 1
    assert result["unmatched_calls"] == 2
    assert result["schema_rejected_submitted_calls"] == 1
    assert result["challenge_score_estimate"] == pytest.approx(0.4)


def test_strict_canonical_matching_is_unordered_but_not_fuzzy() -> None:
    assert score_sample([A, B], [B, A], "hash", "hash")["challenge_score_estimate"] == 1
    gold = [{"name": "x", "arguments": {"a": 1, "b": [1, 2]}}]
    reordered = [{"name": "x", "arguments": {"b": [1, 2], "a": 1}}]
    assert score_sample(gold, reordered, "h", "h")["matched_calls"] == 1
    for changed in (
        {"a": True, "b": [1, 2]}, {"a": 1.0, "b": [1, 2]},
        {"a": 1, "b": [2, 1]}, {"a": 1}, {"a": 1, "b": [1, 2], "extra": 0},
    ):
        assert score_sample(gold, [{"name": "x", "arguments": changed}], "h", "h")["matched_calls"] == 0
    for text in ("Hello", " hello "):
        assert score_sample(
            [{"name": "x", "arguments": {"text": "hello"}}],
            [{"name": "x", "arguments": {"text": text}}], "h", "h",
        )["matched_calls"] == 0


def test_aggregate_includes_failures_and_zero_scores() -> None:
    rows = [
        score_sample([A], [A], "h", "h"),
        score_sample([A, B], [A, B, C], "h", "h"),
        score_sample([A], [A], "bad", "h"),
        score_sample([], None, None, "h"),
    ]
    result = aggregate_scores(rows)
    assert result["total_samples"] == 4
    assert result["challenge_score_estimate"] == 45.0
    assert result["mean_sample_score"] == 0.45
    assert result["perfect_score_samples"] == 1
    assert result["partial_score_samples"] == 1
    assert result["zero_score_samples"] == 2
    assert result["hash_match_samples"] == 2
    assert result["hash_mismatch_samples"] == 1
    assert result["hash_unavailable_samples"] == 1
    assert aggregate_scores([])["challenge_score_estimate"] == 0


def test_reference_hash_uses_only_sdk_full_state(tmp_path) -> None:
    session = FakeSession({"alarm.list"}, state_hash="full-state-with-unchanged-apps")
    factory = FakeFactory(session)
    assert replay_gold_state_hash(tmp_path, "train", "id", [A], session_factory=factory) == (
        "full-state-with-unchanged-apps"
    )
    assert session.calls == [A]
    assert session.validations == [A]
    assert len(factory.creations) == 1


def test_label_loader_never_hashes_partial_final_state_or_creates_sessions(tmp_path, monkeypatch) -> None:
    dataset = _train_dataset(tmp_path)
    monkeypatch.setattr(
        "lifemate.evaluation.replay_gold_state_hash", lambda *_, **__: pytest.fail("loader must not execute gold")
    )
    references = load_label_references(dataset)
    assert references["train_0001"].has_final_state
    assert not hasattr(references["train_0001"], "state_hash")
    raw = json.loads(dataset.input_path.read_text())
    del raw["label"]["final_state"]
    dataset.input_path.write_text(json.dumps(raw) + "\n")
    assert not load_label_references(dataset)["train_0001"].has_final_state


def test_pipeline_fresh_sessions_and_sdk_rejected_trace(tmp_path, monkeypatch) -> None:
    replays = []
    def replay(*args, **kwargs):
        replays.append(args[2])
        return "hash"
    monkeypatch.setattr("lifemate.evaluation.replay_gold_state_hash", replay)
    dataset = _train_dataset(tmp_path)
    references = load_label_references(dataset)

    class FreshFactory:
        def __init__(self):
            self.sessions = []

        def create(self, split, sample_id, dataset_root):
            session = FakeSession(
                {"alarm.list", "sys.ask_missing_info"}, state_hash="hash",
                call=(lambda *_: {"ok": False, "error": "schema rejection"}) if self.sessions else None,
            )
            self.sessions.append(session)
            return session

    factory = FreshFactory()
    runtime = Runtime(dataset, session_factory=factory)
    cache = {}
    def run(name):
        return run_semantic_evaluation(
            dataset=dataset, references=references, selected_ids=("train_0001",), runtime=runtime,
            decision_source_factory=lambda _: ExplicitActionSource([ToolAction("alarm.list", {})]),
            output_dir=tmp_path / name, input_mode="oracle", sample_timeout_seconds=10,
            reference_hash_cache=cache,
        )
    native, constrained = run("native"), run("constrained")
    assert len(factory.sessions) == 2  # Only the independent inference sessions.
    assert replays == ["train_0001"]  # One separate evaluator-owned gold replay, cached for A/B.
    assert factory.sessions[0] is not factory.sessions[1]
    assert factory.sessions[0].calls == factory.sessions[1].calls == [A]
    assert native.aggregate["challenge_score_estimate"] == 100
    assert constrained.aggregate["challenge_score_estimate"] == 0
    row = json.loads(constrained.records_path.read_text())
    assert row["submitted_calls"] == 1
    assert row["unmatched_calls"] == 1
    assert row["schema_rejected_submitted_calls"] == 1
    comparison = compare_evaluations(native, constrained)
    assert comparison["rows"][0]["metric"] == "Challenge score estimate (0–100)"
    assert comparison["sample_transitions"]["regressed_nonzero_to_zero"] == ["train_0001"]
    with pytest.raises(ValueError, match="identical"):
        compare_evaluations(native, replace(constrained, selected_ids=("different",)))


def test_pipeline_failed_sample_remains_in_reports(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("lifemate.evaluation.replay_gold_state_hash", lambda *_, **__: "hash")
    dataset = _train_dataset(tmp_path)
    def fail(_sample):
        raise RuntimeError("no usable predictions")
    summary = run_semantic_evaluation(
        dataset=dataset, references=load_label_references(dataset), selected_ids=("train_0001",),
        runtime=None, decision_source_factory=fail, output_dir=tmp_path / "failed",
        input_mode="oracle", sample_timeout_seconds=10,
    )
    assert summary.aggregate["total_samples"] == 1
    assert summary.aggregate["zero_score_samples"] == 1
    assert summary.aggregate["challenge_score_estimate"] == 0
    assert "unusable_prediction" in summary.report_path.read_text()


def test_paired_transitions_and_markdown(tmp_path) -> None:
    ids = ("improved", "regressed", "perfect", "lost_hash")
    before_scores = (0, 0.8, 0.8, 1)
    after_scores = (0.8, 0, 1, 0)
    def summary(name, scores, hashes):
        rows = [
            {"id": sample_id, **score_sample([A], [A], "h" if match else "bad", "h"),
             "challenge_score_estimate": score}
            for sample_id, score, match in zip(ids, scores, hashes, strict=True)
        ]
        path = tmp_path / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return EvaluationSummary(path, tmp_path / "metrics", tmp_path / "report", ids, aggregate_scores(rows))
    comparison = compare_evaluations(
        summary("native", before_scores, (False, True, True, True)),
        summary("constrained", after_scores, (True, True, True, False)),
    )
    assert comparison["sample_transitions"] == {
        "improved_zero_to_nonzero": ["improved"], "regressed_nonzero_to_zero": ["regressed", "lost_hash"],
        "became_perfect": ["perfect"], "lost_correct_state_hash": ["lost_hash"],
    }
    path = tmp_path / "comparison.md"
    write_comparison_markdown(path, comparison)
    assert "Challenge score estimate" in path.read_text()
    assert "lost_hash" in path.read_text()


def test_diagnostics_separate_controller_errors_from_missing_results() -> None:
    rows = [
        {"id": "abnormal", "metrics": {}, "termination_reason": "max_steps_exhausted",
         "step_statuses": ["schema_rejected", "schema_rejected", "invalid_finish", "model_bad_output"],
         "model_requests": 4, "sample_latency_seconds": 1,
         **score_sample([A], [A], "h", "h")},
        {"id": "failed", "metrics": None, "termination_reason": "sample_timeout",
         "step_statuses": [], "model_requests": 0, "sample_latency_seconds": 2,
         **score_sample([A], None, None, "h")},
    ]
    rows[0]["metrics"] = _compare_calls([A], [A], {}, {})
    result = _aggregate(rows, Counter(), Counter())
    assert result["failure_rate"] == 0.5  # Legacy missing-result metric is retained.
    assert result["infrastructure_failure_rate"] == 0.5
    assert result["abnormal_termination_rate"] == 0.5
    assert result["schema_rejected_proposals"] == 2
    assert result["schema_rejection_sample_rate"] == 0.5
    assert result["invalid_finish_proposals"] == 1
    assert result["malformed_output_samples"] == 1
    assert result["invalid_output_rate"] == 0.5
    assert result["timeout_rate"] == 0.5
    assert result["input_tokens"] == 0
    assert result["output_tokens"] == 0
    assert result["total_tokens"] == 0
    assert result["tokens_per_sample"] == 0


def test_gold_replay_happens_after_inference_without_changing_predictions(tmp_path) -> None:
    dataset = _train_dataset(tmp_path)
    references = load_label_references(dataset)
    inference = FakeFactory(FakeSession({"alarm.list", "sys.ask_missing_info"}, state_hash="full"))
    order = []

    class ReferenceFactory(FakeFactory):
        def create(self, *args):
            assert order == ["inference"]
            assert inference.session.calls == [A]
            order.append("reference")
            return super().create(*args)

    reference = ReferenceFactory(FakeSession({"alarm.list"}, state_hash="full"))
    def decisions(sample):
        assert not hasattr(sample, "label")
        assert not hasattr(sample, "state_hash")
        order.append("inference")
        return ExplicitActionSource([ToolAction("alarm.list", {})])

    summary = run_semantic_evaluation(
        dataset=dataset, references=references, selected_ids=("train_0001",),
        runtime=Runtime(dataset, session_factory=inference), decision_source_factory=decisions,
        output_dir=tmp_path / "isolated", input_mode="oracle", sample_timeout_seconds=10,
        reference_session_factory=reference,
    )
    row = json.loads(summary.records_path.read_text())
    assert order == ["inference", "reference"]
    assert row["hash_match"] is True
    assert row["challenge_score_estimate"] == 1
    assert row["predicted_state_hash"] == "full"
    assert row["predicted_tool_calls"] == inference.session.get_tool_calls() == [A]
    assert reference.session is not inference.session
    assert len(reference.creations) == len(inference.creations) == 1


def test_evaluation_propagates_sample_deadline_to_decision_source(tmp_path) -> None:
    dataset = _train_dataset(tmp_path)
    references = load_label_references(dataset)
    inference = FakeFactory(FakeSession({"alarm.list", "sys.ask_missing_info"}, state_hash="full"))

    class DeadlineSource(ExplicitActionSource):
        deadline = None

        def set_deadline(self, deadline) -> None:
            self.deadline = deadline

    source = DeadlineSource([ToolAction("alarm.list", {})])
    run_semantic_evaluation(
        dataset=dataset,
        references=references,
        selected_ids=("train_0001",),
        runtime=Runtime(dataset, session_factory=inference),
        decision_source_factory=lambda _: source,
        output_dir=tmp_path / "deadline",
        input_mode="oracle",
        sample_timeout_seconds=10,
        compute_reference_state_hash=False,
    )

    assert source.deadline is not None
    assert 0 < source.deadline.remaining_seconds() <= 10


def test_jsonl_writer_serializes_nested_read_only_mappings(tmp_path) -> None:
    path = tmp_path / "records.jsonl"
    _write_jsonl(path, [{"diagnostic": MappingProxyType({"nested": MappingProxyType({"ok": True})})}])

    assert json.loads(path.read_text()) == {"diagnostic": {"nested": {"ok": True}}}


def test_gold_replay_error_is_unavailable_not_hash_mismatch(tmp_path) -> None:
    dataset = _train_dataset(tmp_path)
    references = load_label_references(dataset)
    rejected = FakeFactory(FakeSession({"alarm.list"}, validate=lambda *_: {"ok": False}))
    inference = FakeFactory(FakeSession({"alarm.list", "sys.ask_missing_info"}, state_hash="full"))
    summary = run_semantic_evaluation(
        dataset=dataset, references=references, selected_ids=("train_0001",),
        runtime=Runtime(dataset, session_factory=inference),
        decision_source_factory=lambda _: ExplicitActionSource([ToolAction("alarm.list", {})]),
        output_dir=tmp_path / "reference-error", input_mode="oracle", sample_timeout_seconds=10,
        reference_session_factory=rejected,
    )
    row = json.loads(summary.records_path.read_text())
    assert row["hash_match"] is None
    assert row["expected_state_hash"] is None
    assert "validation failed" in row["reference_state_hash_error"]
    assert row["predicted_state_hash"] == "full"
    assert row["score_reason"] == "gold_or_system_hash_unavailable"
    assert summary.aggregate["hash_mismatch_samples"] == 0
    assert summary.aggregate["reference_hash_failure_samples"] == 1
    assert "reference_error=" in summary.report_path.read_text()


def test_empty_gold_replay_hashes_unchanged_full_state(tmp_path) -> None:
    reference = FakeFactory(FakeSession(set(), state_hash="unchanged-full-state"))
    assert replay_gold_state_hash(tmp_path, "train", "empty", [], session_factory=reference) == "unchanged-full-state"
    assert reference.session.calls == []


@pytest.mark.parametrize("validation,observation", [
    ({"ok": False}, {"ok": True}),
    ({"ok": True}, {"ok": False}),
])
def test_gold_replay_rejections_fail_explicitly(tmp_path, validation, observation) -> None:
    factory = FakeFactory(FakeSession(
        {"alarm.list"}, validate=lambda *_: validation, call=lambda *_: observation,
    ))
    with pytest.raises(ValueError, match="gold replay"):
        replay_gold_state_hash(tmp_path, "train", "id", [A], session_factory=factory)
