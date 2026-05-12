"""JSON Schema helpers for clip manifests (see schemas/dmvc_clip.schema.json)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_PATH = _REPO_ROOT / "schemas" / "dmvc_clip.schema.json"


@lru_cache(maxsize=1)
def load_clip_schema() -> dict[str, Any]:
    """Load the clip JSON Schema bundled in the repository."""
    text = _SCHEMA_PATH.read_text(encoding="utf-8")
    return json.loads(text)


def validate_clip_manifest(instance: dict[str, Any]) -> None:
    """Raise jsonschema.ValidationError if instance does not validate."""
    schema = load_clip_schema()
    Draft202012Validator(schema).validate(instance)


def repo_root() -> Path:
    """Absolute path to repository root (contains schemas/, docs/, etc.)."""
    return _REPO_ROOT
