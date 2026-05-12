#!/usr/bin/env python3
"""Validate JSON syntax for schema and config files used by the PoC."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_DIRS = ("schemas", "configs")


def main() -> int:
    errors: list[str] = []
    for dirname in JSON_DIRS:
        directory = ROOT / dirname
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.json")):
            try:
                text = path.read_text(encoding="utf-8")
                json.loads(text)
            except Exception as exc:  # noqa: BLE001 - surface every failure
                errors.append(f"{path.relative_to(ROOT)}: {exc}")

    if errors:
        print("JSON validation failed:", file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        return 1

    print("All JSON files under schemas/ and configs/ are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
