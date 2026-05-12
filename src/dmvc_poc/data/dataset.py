"""Manifest-backed clip index (Phase 1 minimal; media paths not loaded here)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from dmvc_poc.data.schema import validate_clip_manifest


class ClipManifestIndex:
    """In-memory index over validated clip manifests."""

    def __init__(self, manifests: list[dict[str, Any]]) -> None:
        for item in manifests:
            validate_clip_manifest(item)
        self._manifests = manifests

    def __len__(self) -> int:
        return len(self._manifests)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._manifests)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._manifests[idx]

    @classmethod
    def from_json_file(cls, path: Path | str) -> ClipManifestIndex:
        """Load a JSON file whose root is one manifest object or a list of manifests."""
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, list):
            return cls(data)
        if isinstance(data, dict):
            return cls([data])
        msg = "Root JSON value must be an object or array"
        raise TypeError(msg)
