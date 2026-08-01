import math

from experiments.evaluate_multiseed import _evaluate_route


class EastOnlyPolicy:
    def predict(self, observation, deterministic=True):
        return 3, None


def test_rl_event_is_measured_at_post_move_position():
    scenario = {
        "grid_size": 5,
        "blocked": [],
        "dynamic_obstacles": [
            {
                "cell": [2, 2],
                "period": 1,
                "initial_state": "passable",
            }
        ],
        "stochastic_obstacles": [],
        "moving_obstacles": [],
        "traversal_penalties": [],
        "dynamics_seed": 4,
        "no_fly_cells": [],
        "no_fly_mode": None,
        "no_fly_penalty": 5.0,
        "sensor_noise_probability": 0.0,
        "start": [2, 1],
        "goal": [2, 3],
    }
    variant = {
        "potential_shaping": True,
        "observation_mode": "local",
    }

    route, events = _evaluate_route(
        EastOnlyPolicy(),
        scenario,
        variant,
        "post_move_observed",
    )

    assert route["success"]
    assert route["steps_taken"] == 2
    assert math.isclose(route["route_path_cost"], 2.0)
    assert len(events) == 1
    event = events[0]
    assert event["change_step"] == 1
    assert event["changed_cells_json"] == "[[2, 2]]"
    assert math.isclose(event["pre_change_optimal_cost"], 1.0)
    assert math.isclose(event["post_change_optimal_cost"], 1.0)
    assert event["recovery_steps"] == 0
    assert event["reaction_time_ms"] is not None
