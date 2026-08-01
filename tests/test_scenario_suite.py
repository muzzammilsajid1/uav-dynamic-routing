import json

import pytest

from evaluation.scenario_suite import grid_environment, load_suite


def test_load_suite_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "scenarios": [
                    {"scenario_id": "same"},
                    {"scenario_id": "same"},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique"):
        load_suite(path)


def test_grid_environment_reconstructs_persisted_cells():
    scenario = {
        "grid_size": 5,
        "blocked": [[1, 1], [2, 2]],
        "dynamic_obstacles": [
            {"cell": [3, 3], "period": 4, "initial_state": "blocked"}
        ],
        "start": [0, 0],
        "goal": [4, 4],
    }
    env = grid_environment(scenario)
    assert env.is_blocked((1, 1))
    assert env.is_blocked((3, 3))
    assert env.start == (0, 0)
    assert env.goal == (4, 4)
