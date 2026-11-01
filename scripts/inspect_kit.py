"""Read-only structural check of LifeMate public input and catalog files."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/train+public_test"),
                        help="Directory containing public_test/ (NOT public_test/ itself).")
    args = parser.parse_args()
    root = args.root.resolve()
    split = root / "public_test"
    samples = split / "public_test.jsonl"
    catalog_path = split / "public_test_tools_catalog.json"
    sdk_init = split / "zaic_tools" / "__init__.py"
    required = [root / "README.md", samples, catalog_path, sdk_init]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        print("ERROR: missing expected files:\n" + "\n".join(missing))
        print("Hint: --root must name the folder containing public_test/, not public_test/ itself.")
        return 1

    catalog = json.loads(catalog_path.read_text(encoding="utf-8-sig"))
    apps = catalog.get("apps", [])
    tools = [tool for app in apps for tool in app.get("tools", [])]
    kind = Counter(tool.get("kind", "unknown") for tool in tools)
    extensions = Counter(
        ext for tool in tools for prop in tool.get("parameters", {}).get("properties", {}).values()
        for ext in ("x-ref", "x-match") if ext in prop
    )
    extensions["x-one-of"] = sum("x-one-of" in t.get("parameters", {}) for t in tools)
    print(f"Kit root: {root}")
    version_path = root / "VERSION"
    if version_path.is_file():
        print("Version metadata:")
        for ln in version_path.read_text(encoding="utf-8-sig").splitlines()[:3]:
            print(f"  {ln}")
    print(f"Catalog schema_version: {catalog.get('schema_version', 'unknown')}")
    print(f"Apps: {len(apps)}, tools: {len(tools)}, kinds: {dict(kind)}")
    print(f"Schema extensions (counts): {dict(extensions)}")

    seen = set()
    duplicates = []
    missing_audio = []
    histories = 0
    history_user_audio = 0
    sample_count = 0
    sample_keys = None
    for sample in load_jsonl(samples):
        sample_count += 1
        if sample_keys is None:
            sample_keys = sorted(sample)
        sample_id = sample.get("id")
        if not isinstance(sample_id, str):
            raise ValueError(f"sample #{sample_count} has missing or non-string id")
        if sample_id in seen:
            duplicates.append(sample_id)
        seen.add(sample_id)
        audio_rel = sample.get("audio")
        audio_path = (split / audio_rel).resolve() if isinstance(audio_rel, str) else None
        if audio_path is None or not audio_path.is_relative_to(split.resolve()) or not audio_path.is_file():
            missing_audio.append(sample_id)
        history = sample.get("history") or []
        if history:
            histories += 1
        history_user_audio += sum(
            1 for turn in history if isinstance(turn, dict) and turn.get("role") == "user" and turn.get("audio")
        )
    print(f"Samples: {sample_count}, unique IDs: {len(seen)}")
    print(f"Sample keys: {sample_keys}")
    print(f"Samples with history: {histories}; user history audio refs: {history_user_audio}")
    print(
        f"Missing audio: {len(missing_audio)}; duplicate IDs: {len(duplicates)}"
    )
    categories = (("missing audio", missing_audio), ("duplicate ID", duplicates))
    for category, values in categories:
        if values:
            print(f"  First {category} examples: {values[:5]}")
    print("No environment paths or audio contents were read; train/ was not accessed.")
    return 0 if not (missing_audio or duplicates) else 1


if __name__ == "__main__":
    raise SystemExit(main())
