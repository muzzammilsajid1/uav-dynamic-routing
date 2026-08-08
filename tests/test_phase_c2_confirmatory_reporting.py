import json
from pathlib import Path

import pytest

from scripts.analyze_phase_c2_confirmatory import mean_interval


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "evaluation" / "results"


def test_seed_level_t_interval_uses_independent_training_seeds():
    result = mean_interval([1.0, 0.9375, 239 / 240, 0.9625, 1.0])
    assert result["n_training_seeds"] == 5
    assert result["mean"] == pytest.approx(0.9791666666666666)
    assert result["sample_sd"] == pytest.approx(0.02810570325673342)
    assert result["mean_t_95_ci"] == pytest.approx(
        [0.9442688267885826, 1.0140645065447509]
    )
    assert result["mean_t_95_ci_bounded_0_1"] == pytest.approx(
        [0.9442688267885826, 1.0]
    )


def test_confirmatory_summary_contract():
    payload = json.loads(
        (RESULTS / "phase_c2_v15_m2_confirmatory_summary.json").read_text(encoding="utf-8")
    )
    assert payload["classification"] == "confirmatory_validation_multi_training_seed"
    assert payload["integrity"]["status"] == "passed"
    assert payload["protocol"]["confirmatory_seeds"] == [11, 22, 33, 44, 55]
    assert payload["protocol"]["final_test_status"] == "sealed_not_accessed"
    assert payload["primary_success_rate"]["n_training_seeds"] == 5
    assert payload["descriptive_route_total"]["successes"] == 1175
    assert payload["descriptive_route_total"]["episodes"] == 1200
    assert payload["final_failures"]["collisions"] == 0
    assert payload["final_failures"]["invalid_actions"] == 0
    assert len(payload["runs"]) == 5
    assert all(run["archives"]["latest_bundle"]["zip_integrity"] == "passed"
               for run in payload["runs"])


@pytest.mark.parametrize(
    "name",
    [
        "phase_c2_v15_m2_confirmatory_overview.png",
        "phase_c2_v15_m2_confirmatory_scale_distance.png",
    ],
)
def test_confirmatory_figures_are_nontrivial_png_files(name):
    path = RESULTS / name
    assert path.stat().st_size > 100_000
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
