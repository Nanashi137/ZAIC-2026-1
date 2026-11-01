"""Check public_test submission ID coverage and JSONL structure (NOT a scorer)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: bad JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            yield line_no, item


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/train+public_test"))
    parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.dataset_root / "public_test" / "public_test.jsonl"
    if not input_path.is_file() or not args.submission.is_file():
        print(f"Missing input or submission file: {input_path} / {args.submission}")
        return 1
    expected_ids = []
    for _, item in load_jsonl(input_path):
        sample_id = item.get("id")
        if not isinstance(sample_id, str):
            print("ERROR: input sample missing id")
            return 1
        expected_ids.append(sample_id)
    if len(expected_ids) != len(set(expected_ids)):
        print("ERROR: duplicate input sample IDs")
        return 1

    ids = []
    errors = []
    for line_no, row in load_jsonl(args.submission):
        sid = row.get("id")
        ids.append(sid)
        if set(row) != {"id", "tool_calls", "state_hash"}:
            errors.append(f"line {line_no}: expected exactly id/tool_calls/state_hash")
        if not isinstance(sid, str):
            errors.append(f"line {line_no}: id must be string")
        calls = row.get("tool_calls")
        if not isinstance(calls, list):
            errors.append(f"line {line_no}: tool_calls must be array")
        else:
            for i, call in enumerate(calls):
                valid_call = (
                    isinstance(call, dict)
                    and set(call) == {"name", "arguments"}
                    and isinstance(call.get("name"), str)
                    and isinstance(call.get("arguments"), dict)
                )
                if not valid_call:
                    errors.append(f"line {line_no}: tool_calls[{i}] must contain name:string, arguments:object")
        if not isinstance(row.get("state_hash"), str) or not row.get("state_hash"):
            errors.append(f"line {line_no}: state_hash must be non-empty string returned by SDK")
    unique = set(ids)
    missing = set(expected_ids) - unique
    extra = unique - set(expected_ids)
    if len(unique) != len(ids):
        errors.append(f"duplicate submission IDs: {len(ids) - len(unique)}")
    if missing:
        errors.append(f"missing IDs ({len(missing)}): {sorted(missing)[:5]}")
    if extra:
        errors.append(f"extra IDs ({len(extra)}): {sorted(str(x) for x in extra)[:5]}")
    print(f"Expected samples: {len(expected_ids)}, submission rows: {len(ids)}")
    if errors:
        print("STRUCTURE CHECK FAILED:")
        for message in errors[:25]:
            print(" -", message)
        if len(errors) > 25:
            print(f" ... {len(errors) - 25} more")
        return 1
    print("PASS: submission is structurally valid and covers every expected ID.")
    print("Not checked: official tool schemas, true SDK state hashes, user intent, or task success.")
    print("Next: run the official 'python -m zaic_tools ... --validate <path>' command.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
