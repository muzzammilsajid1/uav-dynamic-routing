from __future__ import annotations

import pytest

from evaluation.summarize_timing_distributions import summarize


def test_timing_summary_reports_repetitions_median_and_std() -> None:
    rows = [
        {
            "method": "astar",
            "split": "scale_15",
            "policy_seed": "classical",
            "scenario_id": "s1",
            "repetition": str(repetition),
            "route_compute_time_ms": str(value),
            "mean_decision_latency_ms": str(value / 2),
        }
        for repetition, value in enumerate([1.0, 2.0, 3.0], start=1)
    ]
    result = summarize(rows)
    assert len(result) == 1
    assert result[0]["repetitions_per_scenario"] == 3
    assert result[0]["route_median_ms"] == pytest.approx(2.0)
    assert result[0]["route_sample_std_ms"] == pytest.approx(1.0)


def test_rl_timing_summary_includes_seed_and_aggregate_rows() -> None:
    rows = [
        {
            "method": "rl_full",
            "split": "scale_15",
            "policy_seed": seed,
            "scenario_id": "s1",
            "repetition": "1",
            "route_compute_time_ms": value,
            "mean_decision_latency_ms": value,
        }
        for seed, value in [("11", "1.0"), ("22", "2.0")]
    ]
    result = summarize(rows)
    assert {row["policy_seed"] for row in result} == {"11", "22", "all_seeds"}
