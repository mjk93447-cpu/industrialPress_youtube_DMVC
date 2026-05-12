"""Manifest-backed clip index and training-eligibility gate checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from dmvc_poc.data.rights import ensure_training_eligible
from dmvc_poc.data.schema import validate_clip_manifest


class ClipManifestIndex:
    """In-memory index over validated clip manifests."""

    def __init__(
        self,
        manifests: list[dict[str, Any]],
        *,
        enforce_training_eligibility: bool = True,
    ) -> None:
        for item in manifests:
            validate_clip_manifest(item)
            if enforce_training_eligibility:
                ensure_training_eligible(item)
        self._manifests = manifests
        self._enforce_training_eligibility = enforce_training_eligibility

    def __len__(self) -> int:
        return len(self._manifests)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return iter(self._manifests)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._manifests[idx]

    @classmethod
    def from_json_file(
        cls,
        path: Path | str,
        *,
        enforce_training_eligibility: bool = True,
    ) -> ClipManifestIndex:
        """Load a JSON file whose root is one manifest object or a list of manifests."""
        raw = Path(path).read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, list):
            return cls(data, enforce_training_eligibility=enforce_training_eligibility)
        if isinstance(data, dict):
            return cls([data], enforce_training_eligibility=enforce_training_eligibility)
        msg = "Root JSON value must be an object or array"
        raise TypeError(msg)


class ProcessedClipDataset:
    """Read preprocessed ``.npz`` clip arrays by clip_id."""

    def __init__(self, manifests: list[dict[str, Any]], processed_root: Path | str) -> None:
        self._manifests = manifests
        self._processed_root = Path(processed_root)

    def __len__(self) -> int:
        return len(self._manifests)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        row = self._manifests[idx]
        clip_id = str(row["clip_id"])
        path = self._processed_root / f"{clip_id}.npz"
        if not path.is_file():
            msg = f"Processed file missing for clip_id={clip_id}: {path}"
            raise FileNotFoundError(msg)
        with np.load(path) as data:
            return {
                "video": data["video"].astype(np.float32),
                "audio": data["audio"].astype(np.float32),
                "frame_diff": data["frame_diff"].astype(np.float32),
            }
