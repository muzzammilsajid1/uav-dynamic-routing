import json
from pathlib import Path

import pytest

from scripts.compare_phase_c2_models import exact_mcnemar_p
from scripts.summarize_phase_c2 import wilson


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"


def test_route_level_interval_and_exact_paired_test():
    lower, upper = wilson(240, 240)
    assert lower == pytest.approx(0.9842460800584412)
    assert upper == pytest.approx(1.0)
    assert exact_mcnemar_p(0, 40) == pytest.approx(1.8189894035458565e-12)
    assert exact_mcnemar_p(0, 0) == 1.0


def test_m2_development_summary_contract():
    payload = json.loads(
        (RESULTS / "phase_c2_v12_m2_seed42_summary.json").read_text(encoding="utf-8")
    )
    assert payload["classification"] == "development_validation_single_seed"
    assert len(payload["checkpoints"]) == 8
    assert payload["final_checkpoint"] == 301056
    assert payload["final_checkpoint_metrics"]["successes"] == 240
    assert payload["final_checkpoint_metrics"]["collisions"] == 0
    assert payload["checkpoints"][-1]["invalid_actions"] == 0


def test_m1_m2_comparison_is_limited_and_paired():
    payload = json.loads(
        (RESULTS / "phase_c2_v12_m1_m2_seed42_comparison.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["classification"] == "paired_development_validation_single_training_seed"
    assert payload["identical_route_order"] is True
    assert "training-seed variability" in payload["claim_limit"]
    paired = payload["paired_final_comparison"]
    assert paired["m1_only_successes"] == 0
    assert paired["m2_only_successes"] == 40
    assert payload["models"]["M1"]["trainable_parameters"] == 428937
    assert payload["models"]["M2"]["trainable_parameters"] == 9545


@pytest.mark.parametrize(
    "name",
    [
        "phase_c2_v12_m2_seed42_learning_curve.png",
        "phase_c2_v12_m1_m2_seed42_comparison.png",
    ],
)
def test_phase_c2_figures_are_png_files(name):
    path = RESULTS / name
    assert path.stat().st_size > 100_000
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
