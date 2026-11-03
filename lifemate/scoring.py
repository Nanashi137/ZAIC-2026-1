"""Offline challenge-score estimate; the kit does not ship its call matcher.

Matching is order-independent multiset intersection of exact names and canonical
JSON arguments. Dict key order is ignored; array order, JSON scalar types, case,
whitespace and all argument fields are significant. No fuzzy/partial matching.
SDK-rejected submitted calls cannot match; local pre-dispatch rejections are not
submitted calls. Gold and submitted lists are never modified.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sdk import OfficialSessionFactory, SessionFactory

MATCHING_METHOD = "strict-canonical-json-unordered-one-to-one-v1"
REFERENCE_HASH_SOURCE = "sdk-gold-replay-full-state-v1"


def replay_gold_state_hash(
    dataset_root: Path,
    split: str,
    sample_id: str,
    gold_calls: Sequence[Mapping[str, Any]],
    *,
    session_factory: SessionFactory | None = None,
) -> str:
    """Offline reference execution, never an inference session or planning input.

    label.final_state contains changed apps ONLY and cannot be hashed against a
    full submission state. Replay labeled calls in a separate official session
    and obtain its authoritative full-state hash, including unchanged apps.
    No direct state reads, get_state(), inference callbacks, or retries occur.
    """
    session = (session_factory or OfficialSessionFactory()).create(split, sample_id, dataset_root)
    for index, call in enumerate(gold_calls):
        name, arguments = str(call["name"]), dict(call.get("arguments", {}))
        if session.validate(name, arguments).get("ok") is not True:
            raise ValueError(f"gold replay validation failed at call {index}")
        if session.call(name, arguments).get("ok") is not True:
            raise ValueError(f"gold replay SDK rejected call {index}")
    value = session.get_state_hash()
    if not isinstance(value, str) or not value:
        raise ValueError("gold replay did not produce a usable SDK state hash")
    return value


def _key(call: Mapping[str, Any]) -> str:
    return json.dumps(
        {"name": call["name"], "arguments": call.get("arguments", {})},
        sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False,
    )


def score_sample(
    gold: Sequence[Mapping[str, Any]],
    submitted: Sequence[Mapping[str, Any]] | None,
    system_hash: str | None,
    gold_hash: str | None,
    *,
    rejected_indices: Sequence[int] = (),
) -> dict[str, Any]:
    """Apply the supplied formula with explicitly approximate call matching.

    Missing gold hashes yield zero for nonempty gold (unknown, never presumed
    equal). Empty-gold scoring ignores hashes. No usable prediction always scores
    zero, including empty gold. Denominators include all submitted SDK attempts.
    """
    remaining = Counter(_key(call) for call in gold)
    rejected = set(rejected_indices)
    matched = 0
    for index, call in enumerate(submitted or ()):
        key = _key(call)
        if index not in rejected and remaining[key] > 0:
            remaining[key] -= 1
            matched += 1
    count = len(submitted) if submitted is not None else 0
    unmatched = count - matched
    precision = matched / count if count else 0.0
    recall = matched / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hash_match = system_hash == gold_hash if system_hash is not None and gold_hash is not None else None
    if submitted is None:
        score, reason = 0.0, "unusable_prediction"
    elif not gold:
        score = 1.0 if not submitted else 0.0
        reason = "empty_gold_empty_submission" if not submitted else "empty_gold_nonempty_submission"
    elif hash_match is not True:
        score = 0.0
        reason = "hash_mismatch" if hash_match is False else "gold_or_system_hash_unavailable"
    elif matched == 0:
        score, reason = 0.0, "no_matched_calls"
    else:
        score, reason = f1, "matched_f1"
    return {
        "challenge_score_estimate": score,
        "challenge_score_exact": False,
        "matching_method": MATCHING_METHOD,
        "hash_match": hash_match,
        "gold_calls": len(gold), "submitted_calls": count,
        "matched_calls": matched, "unmatched_calls": unmatched,
        "schema_rejected_submitted_calls": len(rejected.intersection(range(count))),
        "precision": precision, "recall": recall, "f1": f1,
        "score_reason": reason,
    }


def aggregate_scores(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """All evaluated records, including unusable predictions, enter the mean."""
    scores = [float(row["challenge_score_estimate"]) for row in records]
    total = len(records)
    mean = sum(scores) / total if total else 0.0
    reasons = Counter(str(row["score_reason"]) for row in records)
    hashes = Counter(row.get("hash_match") for row in records)
    perfect = sum(score == 1.0 for score in scores)
    partial = sum(0.0 < score < 1.0 for score in scores)
    return {
        "challenge_score_estimate": 100.0 * mean,
        "challenge_score_exact": False,
        "matching_method": MATCHING_METHOD,
        "total_samples": total,
        "mean_sample_score": mean,
        "perfect_score_samples": perfect,
        "partial_score_samples": partial,
        "zero_score_samples": sum(score == 0.0 for score in scores),
        "perfect_sample_rate": perfect / total if total else 0.0,
        "partial_sample_rate": partial / total if total else 0.0,
        "hash_match_samples": hashes[True],
        "hash_mismatch_samples": hashes[False],
        "hash_unavailable_samples": hashes[None],
        "hash_match_rate": hashes[True] / total if total else 0.0,
        "score_reasons": dict(sorted(reasons.items())),
        "zero_due_to_hash_mismatch": sum(
            row["challenge_score_estimate"] == 0 and row["gold_calls"] > 0
            and row.get("hash_match") is False for row in records
        ),
        "zero_due_to_no_matched_calls": sum(
            row["challenge_score_estimate"] == 0 and row["gold_calls"] > 0
            and row["matched_calls"] == 0 for row in records
        ),
        "empty_gold_samples": sum(row["gold_calls"] == 0 for row in records),
        "empty_gold_perfect_samples": reasons["empty_gold_empty_submission"],
        "empty_gold_nonempty_submission_samples": reasons["empty_gold_nonempty_submission"],
        **{key: sum(int(row.get(key, 0)) for row in records) for key in (
            "gold_calls", "submitted_calls", "matched_calls", "unmatched_calls",
            "schema_rejected_submitted_calls",
        )},
    }