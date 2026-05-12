from __future__ import annotations

import pytest

from dmvc_poc.train.smoke import run_smoke_training_step


@pytest.mark.smoke
def test_smoke_training_step_cpu_loss_finite() -> None:
    metrics = run_smoke_training_step(device="cpu")
    assert metrics["loss"] == pytest.approx(metrics["loss"])  # not nan
    assert metrics["loss"] > 0.0
