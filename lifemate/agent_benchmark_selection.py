"""Create a reproducible agent confirmation fold disjoint from the first benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .data import DatasetReader
from .evaluation import LabelReference, load_label_references, select_sample_ids


def select_disjoint_ids(
    references: Mapping[str, LabelReference],
    *,
    count: int,
    seed: int,
    excluded_count: int,
    excluded_seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if count < 1 or excluded_count < 1:
        raise ValueError("count and excluded_count must be positive")
    excluded = select_sample_ids(references, sample_ids=None, limit=excluded_count, seed=excluded_seed)
    excluded_set = set(excluded)
    candidates = {sample_id: reference for sample_id, reference in references.items() if sample_id not in excluded_set}
    if len(candidates) < count:
        raise ValueError(
            f"only {len(candidates)} samples remain after excluding {len(excluded)}; cannot select {count}"
        )
    selected = select_sample_ids(candidates, sample_ids=None, limit=count, seed=seed)
    if excluded_set.intersection(selected):
        raise AssertionError("confirmation fold overlaps the excluded benchmark fold")
    return selected, excluded


def materialize_confirmation_fold(
    *,
    dataset_root: Path,
    output_root: Path,
    count: int = 100,
    seed: int = 314159,
    excluded_count: int = 100,
    excluded_seed: int = 42,
) -> tuple[Path, Path]:
    dataset = DatasetReader(dataset_root, "train", allow_train=True)
    references = load_label_references(dataset)
    selected, excluded = select_disjoint_ids(
        references,
        count=count,
        seed=seed,
        excluded_count=excluded_count,
        excluded_seed=excluded_seed,
    )
    selected_text = "".join(f"{sample_id}\n" for sample_id in selected)
    selected_hash = hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest()
    excluded_hash = hashlib.sha256("\n".join(excluded).encode("utf-8")).hexdigest()
    config: dict[str, Any] = {
        "selection_schema_version": 1,
        "dataset_root": str(dataset.dataset_root),
        "split": "train",
        "selected_count": len(selected),
        "selected_seed": seed,
        "selected_ids_sha256": selected_hash,
        "excluded_count": len(excluded),
        "excluded_seed": excluded_seed,
        "excluded_ids_sha256": excluded_hash,
        "overlap_count": 0,
    }
    config_text = json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ids_path = output_root / "selected-ids.txt"
    config_path = output_root / "selection-config.json"
    if ids_path.exists() or config_path.exists():
        if not ids_path.is_file() or not config_path.is_file():
            raise ValueError("confirmation selection is incomplete; expected both selection files")
        if ids_path.read_text(encoding="utf-8") != selected_text:
            raise ValueError("existing selected-ids.txt does not match the requested confirmation fold")
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise ValueError("existing selection-config.json does not match the requested confirmation fold")
        return ids_path, config_path
    ids_path.write_text(selected_text, encoding="utf-8")
    config_path.write_text(config_text, encoding="utf-8")
    return ids_path, config_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/train+public_test"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--excluded-count", type=int, default=100)
    parser.add_argument("--excluded-seed", type=int, default=42)
    args = parser.parse_args(argv)
    ids_path, config_path = materialize_confirmation_fold(
        dataset_root=args.dataset_root,
        output_root=args.output_root,
        count=args.count,
        seed=args.seed,
        excluded_count=args.excluded_count,
        excluded_seed=args.excluded_seed,
    )
    print(json.dumps({"selected_ids": str(ids_path), "config": str(config_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
