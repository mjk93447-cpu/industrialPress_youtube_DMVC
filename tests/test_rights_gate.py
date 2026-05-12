from __future__ import annotations

import pytest

from dmvc_poc.data.rights import assert_valid_transition, ensure_training_eligible


def test_rights_transition_rejects_invalid_jump() -> None:
    with pytest.raises(ValueError):
        assert_valid_transition("needs_relevance_review", "creator_permission_pending")


def test_rights_transition_accepts_expected_path() -> None:
    assert_valid_transition("standard_only_queue", "creator_permission_pending")


def test_training_eligibility_blocks_non_approved_status() -> None:
    manifest = {"rights_status": "needs_relevance_review"}
    with pytest.raises(ValueError):
        ensure_training_eligible(manifest)
