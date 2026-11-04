from __future__ import annotations

from lifemate.agent_benchmark_selection import select_disjoint_ids
from lifemate.evaluation import LabelReference


def _references(count: int) -> dict[str, LabelReference]:
    return {
        f"train_{index:04d}": LabelReference(
            sample_id=f"train_{index:04d}",
            transcript="fixture",
            tool_calls=(),
            tool_results=(),
            has_final_state=True,
        )
        for index in range(count)
    }


def test_confirmation_selection_is_deterministic_and_disjoint() -> None:
    references = _references(300)

    selected, excluded = select_disjoint_ids(
        references,
        count=100,
        seed=314159,
        excluded_count=100,
        excluded_seed=42,
    )
    repeated, repeated_excluded = select_disjoint_ids(
        references,
        count=100,
        seed=314159,
        excluded_count=100,
        excluded_seed=42,
    )

    assert selected == repeated
    assert excluded == repeated_excluded
    assert len(selected) == len(set(selected)) == 100
    assert len(excluded) == len(set(excluded)) == 100
    assert not set(selected).intersection(excluded)
