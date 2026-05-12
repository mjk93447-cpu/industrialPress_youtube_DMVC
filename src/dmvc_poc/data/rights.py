"""Rights transition and training eligibility gate checks."""

from __future__ import annotations

from typing import Any

TRAINING_ALLOWED_RIGHTS_STATUS = {
    "creative_commons",
    "creator_permission",
    "owned_capture",
    "licensed_source",
    "synthetic",
}

_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "needs_relevance_review": {
        "needs_relevance_review",
        "rejected",
        "standard_only_queue",
        "approved_for_training",
    },
    "standard_only_queue": {
        "standard_only_queue",
        "creator_permission_pending",
        "rejected",
        "approved_for_training",
    },
    "creator_permission_pending": {
        "creator_permission_pending",
        "approved_for_training",
        "rejected",
    },
    "approved_for_training": {"approved_for_training", "revoked", "rejected"},
    "rejected": {"rejected"},
    "revoked": {"revoked"},
}


def assert_valid_transition(old: str, new: str) -> None:
    """Validate candidate rights-state transition."""
    allowed = _ALLOWED_TRANSITIONS.get(old)
    if allowed is None:
        msg = f"Unknown source rights status: {old!r}"
        raise ValueError(msg)
    if new not in allowed:
        msg = f"Invalid transition: {old!r} -> {new!r}"
        raise ValueError(msg)


def ensure_training_eligible(manifest: dict[str, Any]) -> None:
    """Hard-fail if a clip manifest is not eligible for training."""
    rights_status = str(manifest.get("rights_status", ""))
    if rights_status not in TRAINING_ALLOWED_RIGHTS_STATUS:
        msg = (
            "Training blocked: rights_status must be one of "
            f"{sorted(TRAINING_ALLOWED_RIGHTS_STATUS)}, got {rights_status!r}"
        )
        raise ValueError(msg)
