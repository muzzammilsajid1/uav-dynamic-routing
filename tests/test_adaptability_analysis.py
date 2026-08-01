from __future__ import annotations

import pytest

from evaluation.analyze_adaptability import (
    _change_signature,
    _paired_tests,
    _summarize,
)


def _event(
    method: str,
    *,
    seed: str,
    scenario_id: str,
    success: bool,
    recovered: bool,
) -> dict[str, object]:
    return {
        "method": method,
        "seed": seed,
        "scenario_id": scenario_id,
        "split": "changed_toggle_periods",
        "change_step": 4,
        "change_signature": "[[3,4]]",
        "post_change_success": success,
        "recovered": recovered,
        "finite_post_cost": True,
        "recovery_steps": 2.0 if recovered else float("nan"),
        "optimal_cost_shock": 1.5,
        "extra_optimal_cost": 1.5,
        "reaction_time_ms": 0.2,
        "node_expansions": 12.0 if seed == "classical" else float("nan"),
    }


def test_change_signature_matches_classical_and_rl_encodings() -> None:
    classical = {
        "change_reason": "dynamic_change:[(3, 4), (1, 2)]",
        "changed_cells_json": "",
    }
    rl = {
        "change_reason": "",
        "changed_cells_json": "[[1, 2], [3, 4]]",
    }
    assert _change_signature(classical) == _change_signature(rl)


def test_adaptability_summary_uses_correct_independent_units() -> None:
    records = [
        _event(
            "astar",
            seed="classical",
            scenario_id="s1",
            success=True,
            recovered=True,
        ),
        _event(
            "astar",
            seed="classical",
            scenario_id="s2",
            success=False,
            recovered=False,
        ),
        _event(
            "rl_full",
            seed="11",
            scenario_id="s1",
            success=True,
            recovered=True,
        ),
        _event(
            "rl_full",
            seed="22",
            scenario_id="s1",
            success=True,
            recovered=False,
        ),
    ]
    summary = _summarize(records)
    astar = next(
        row
        for row in summary
        if row["method"] == "astar" and row["split"] == "all_dynamic"
    )
    rl = next(
        row
        for row in summary
        if row["method"] == "rl_full" and row["split"] == "all_dynamic"
    )
    assert astar["independent_units"] == 2
    assert astar["post_change_success_mean"] == pytest.approx(0.5)
    assert rl["independent_units"] == 2
    assert rl["recovery_rate_mean"] == pytest.approx(0.5)


def test_adaptability_tests_pair_on_change_signature() -> None:
    records = [
        _event(
            "astar",
            seed="classical",
            scenario_id="s1",
            success=True,
            recovered=True,
        ),
        _event(
            "dijkstra",
            seed="classical",
            scenario_id="s1",
            success=False,
            recovered=False,
        ),
    ]
    tests = _paired_tests(records)
    success = next(row for row in tests if row["metric"] == "post_change_success")
    assert success["n"] == 1
    assert success["difference_right_minus_left"] == pytest.approx(-1.0)
