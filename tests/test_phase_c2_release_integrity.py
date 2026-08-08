import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from rl_v3.phase_c2_env import PhaseC2EndpointGenerator, PhaseC2Env
from rl_v3.run_phase_c2 import PhaseC2Runner, _aggregate_validation


ROOT = Path(__file__).resolve().parents[1]


class _CrashEnv:
    def __init__(self, config, mode, generator):
        self.generator = generator
        self.action_space = gym.spaces.Discrete(8)
        self.observation_space = gym.spaces.Dict(
            {"scalars": gym.spaces.Box(-1.0, 7.0, shape=(4,), dtype=np.float32)}
        )
        self._v2 = type(
            "Native",
            (),
            {
                "ACTION_DELTAS": np.asarray(
                    [[-1, 0], [1, 0], [0, -1], [0, 1], [-1, -1], [-1, 1], [1, -1], [1, 1]]
                ),
                "uav_pos": np.asarray([0, 0]),
            },
        )()
        self.unwrapped = self

    def reset(self):
        self._v2.uav_pos = np.asarray([0, 0])
        return {"scalars": np.zeros(4, dtype=np.float32)}, {
            "start": [0, 0],
            "goal": [0, 1],
            "grid_size": 15,
            "budget": 1,
        }

    def action_masks(self):
        return np.asarray([False, True, False, True, False, False, False, True])

    def step(self, action):
        return (
            {"scalars": np.zeros(4, dtype=np.float32)},
            -1.0,
            True,
            False,
            {"crashed": True, "is_success": False, "grid_size": 15},
        )

    def close(self):
        pass


class _RecordingModel:
    n_steps = 2048
    n_envs = 1

    def __init__(self):
        self.masks = []

    def predict(self, observation, deterministic, action_masks):
        self.masks.append(np.asarray(action_masks, dtype=bool))
        return 3, None

    def save(self, path):
        Path(path).write_bytes(b"model")


def test_validation_applies_masks_and_uses_native_crash_field(tmp_path, monkeypatch):
    import rl_v3.run_phase_c2 as module

    monkeypatch.setattr(module, "PhaseC2Env", _CrashEnv)
    runner = PhaseC2Runner.__new__(PhaseC2Runner)
    runner.config = {"reward": {"type": "none"}}
    runner.out_dir = tmp_path
    runner.model_type = "M1"
    runner.model = _RecordingModel()
    runner.generator = type(
        "Generator",
        (),
        {
            "val_manifest": [{"distance_bin": "short"}],
            "val_hash": "manifest-hash",
            "get_state": lambda self: {"active_sizes": [15]},
        },
    )()
    runner.history = {}
    runner.seed = 42
    runner.deterministic_cuda = False

    runner.evaluate_and_save(2048, requested_ts=10)

    assert len(runner.model.masks) == 1
    assert runner.model.masks[0][3]
    artifact = json.loads((tmp_path / "evaluation_002048.json").read_text())
    assert artifact["action_masking_applied"] is True
    assert artifact["collision_field"] == "crashed"
    assert artifact["invalid_action_count"] == 0
    assert artifact["aggregates"]["all"]["collisions"] == 1
    assert artifact["episodes"][0]["failure_label"] == "collision"
    assert artifact["requested_interactions"] == 10
    assert artifact["completed_interactions"] == 2048
    assert (tmp_path / "rng_002048.pt").exists()


def test_validation_aggregates_by_scale_and_distance():
    rows = []
    for size, bucket, success in [(15, "short", True), (30, "long", False)]:
        rows.append(
            {
                "grid_size": size,
                "distance_bin": bucket,
                "is_success": success,
                "crashed": not success,
                "is_timeout": False,
                "two_cell_oscillation": False,
                "longer_loop": False,
                "decisions": 1,
                "episode_return": 1.0 if success else -1.0,
                "final_octile_distance": 0.0 if success else 1.0,
                "path_cost_gap": 0.0 if success else None,
                "path_cost_ratio": 1.0 if success else None,
                "mean_policy_inference_latency_ms": 0.1,
                "mean_masked_action_latency_ms": 0.2,
            }
        )
    aggregates = _aggregate_validation(rows)
    assert aggregates["all"]["episodes"] == 2
    assert aggregates["scale/15"]["success_rate"] == 1.0
    assert aggregates["scale_distance/30/long"]["collisions"] == 1


def test_generator_state_restores_curriculum_sizes():
    generator = PhaseC2EndpointGenerator(seed=42)
    generator.set_active_sizes([15, 30, 50])
    state = generator.get_state()
    restored = PhaseC2EndpointGenerator(seed=7)
    restored.set_active_sizes([100])
    restored.set_state(state)
    assert restored.active_sizes == [15, 30, 50]


def test_rng_state_round_trip(tmp_path):
    path = tmp_path / "rng.pt"
    random.seed(11)
    np.random.seed(11)
    torch.manual_seed(11)
    PhaseC2Runner.save_rng_state(PhaseC2Runner.__new__(PhaseC2Runner), path)
    expected = (random.random(), np.random.random(), torch.rand(1).item())
    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    PhaseC2Runner.restore_rng_state(path)
    actual = (random.random(), np.random.random(), torch.rand(1).item())
    assert actual == expected


def test_interaction_plan_is_ppo_update_aligned(tmp_path):
    runner = PhaseC2Runner(
        ROOT / "configs/rl_v3_phase_c2.json", tmp_path, model_type="M2", device="cpu"
    )
    assert runner.rollout_size() == 2048
    assert runner.checkpoint_plan(150000) == [
        (25000, 26624),
        (50000, 51200),
        (75000, 75776),
        (100000, 100352),
        (150000, 151552),
    ]


def test_phase_c2_reset_constructs_native_environment_once(monkeypatch):
    config = json.loads((ROOT / "configs/rl_v3_phase_c2.json").read_text())
    generator = PhaseC2EndpointGenerator(seed=42)
    generator.val_manifest = [
        {"grid_size": 15, "distance_bin": "short", "start": [0, 0], "goal": [0, 1]}
    ]
    env = PhaseC2Env(config, mode="eval", generator=generator)
    original = env._make_v2
    calls = 0

    def counted():
        nonlocal calls
        calls += 1
        return original()

    monkeypatch.setattr(env, "_make_v2", counted)
    env.reset()
    assert calls == 1
    env.close()


def test_scalar_feature_metadata_matches_implementation():
    config = json.loads((ROOT / "configs/rl_v3_phase_c2.json").read_text())
    assert config["observation"]["scalar_features"] == [
        "relative_goal_row",
        "relative_goal_column",
        "grid_size",
        "previous_action",
    ]
    assert "step_penalty" not in config["reward"]
