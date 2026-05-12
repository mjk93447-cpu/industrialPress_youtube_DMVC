#!/usr/bin/env python3
"""Print roadmap completion table (machine-readable checklist vs repo files)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from dmvc_poc.roadmap.status import compute_completion


def main() -> int:
    parser = argparse.ArgumentParser(description="Roadmap completion report")
    parser.add_argument(
        "--root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root",
    )
    args = parser.parse_args()
    repo_root: Path = args.root

    ratio, rows = compute_completion(repo_root)
    print("# Roadmap completion")
    print()
    print(f"- Required completion ratio: **{ratio:.0%}**")
    print()
    print("| Phase | Id | OK | Description | Paths |")
    print("| --- | --- | --- | --- | --- |")
    for item, ok in rows:
        paths = "<br>".join(item.paths)
        opt = "*(optional)* " if item.optional else ""
        print(
            f"| {item.phase} | {item.item_id} | {'yes' if ok else 'no'} | "
            f"{opt}{item.description} | {paths} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
