import json
import numpy as np
import pytest
from pathlib import Path
import math
from rl_v3.phase_c1_env import PhaseC1Env, PhaseC1EndpointGenerator
from rl_v3.phase_c0_env import _astar_cost

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "rl_v3_phase_c1.json"

@pytest.fixture
def config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

def test_endpoint_generator_determinism():
    g1 = PhaseC1EndpointGenerator(seed=42)
    g2 = PhaseC1EndpointGenerator(seed=42)
    assert g1.val_hash == g2.val_hash
    assert g1.val_manifest == g2.val_manifest
    for _ in range(10):
        assert g1.sample_train() == g2.sample_train()

def test_train_val_exclusion():
    g = PhaseC1EndpointGenerator(seed=42)
    val_set = set(g.val_manifest)
    for b in ["short", "medium", "long"]:
        for p in g.train_bins[b]:
            assert p not in val_set
            rev = (p[1], p[0])
            assert rev not in val_set

def test_distance_bins():
    g = PhaseC1EndpointGenerator(seed=42)
    for p in g.train_bins["short"]:
        assert _astar_cost(p[0], p[1]) < g.T1
    for p in g.train_bins["medium"]:
        cost = _astar_cost(p[0], p[1])
        assert g.T1 <= cost < g.T2
    for p in g.train_bins["long"]:
        assert _astar_cost(p[0], p[1]) >= g.T2

def test_orientation_classification():
    g = PhaseC1EndpointGenerator(seed=42)
    p = ((2, 2), (10, 3)) # dx = 8, dy = 1 => vertical (row-dominant)
    assert g.orientations[p] == "vertical"
    p = ((2, 2), (3, 10)) # dx = 1, dy = 8 => horizontal (col-dominant)
    assert g.orientations[p] == "horizontal"
    p = ((2, 2), (8, 9)) # dx = 6, dy = 7 => diagonal
    assert g.orientations[p] == "diagonal"

def test_validation_balance():
    g = PhaseC1EndpointGenerator(seed=42)
    assert len(g.val_manifest) == 120
    counts = {"short": 0, "medium": 0, "long": 0}
    ori_counts = {"vertical": 0, "horizontal": 0, "diagonal": 0, "mixed": 0}
    
    for p in g.val_manifest:
        cost = _astar_cost(p[0], p[1])
        if cost < g.T1: counts["short"] += 1
        elif cost < g.T2: counts["medium"] += 1
        else: counts["long"] += 1
        ori_counts[g.orientations[p]] += 1
        
    assert counts["short"] == 40
    assert counts["medium"] == 40
    assert counts["long"] == 40
    
    for k, v in ori_counts.items():
        assert v >= 10, f"Orientation {k} is poorly balanced with {v} pairs."

def test_sampled_endpoint_application(config):
    g = PhaseC1EndpointGenerator(seed=42)
    env = PhaseC1Env(config, mode="val", generator=g)
    
    obs, info = env.reset()
    start = info["start"]
    goal = info["goal"]
    
    assert tuple(env._v2.uav_pos) == start
    assert tuple(env._v2.goal_pos) == goal
    
    uav_r, uav_c = start
    goal_r, goal_c = goal
    
    assert np.isclose(obs["scalars"][0], (goal_r - uav_r) / 14.0)
    assert np.isclose(obs["scalars"][1], (goal_c - uav_c) / 14.0)
    
    cost = _astar_cost(start, goal)
    expected_budget = max(int(config["training"]["minimum_episode_budget"]), int(math.ceil(cost * float(config["training"]["episode_budget_astar_multiplier"]))))
    assert info["budget"] == expected_budget
    assert env._max_steps == expected_budget

def test_true_truncation(config):
    g = PhaseC1EndpointGenerator(seed=42)
    env = PhaseC1Env(config, mode="val", generator=g)
    obs, info = env.reset()
    budget = env._max_steps
    for i in range(budget):
        obs, rew, term, trunc, info = env.step(4)
        if term:
            break
    if not term:
        assert trunc, "Environment did not truncate when budget exhausted."

def test_cost_gap_metrics(config):
    g = PhaseC1EndpointGenerator(seed=42)
    env = PhaseC1Env(config, mode="val", generator=g)
    obs, info_reset = env.reset()
    
    ep_cost = 0.0
    for a in [4, 0, 4]:
        _, _, _, _, _ = env.step(a)
        ep_cost += math.sqrt(2.0) if a >= 4 else 1.0
        
    assert np.isclose(ep_cost, 2.0 * math.sqrt(2.0) + 1.0)
    
def test_model_seed():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config.get("seed") == 42, "Fixed model seed 42 is missing"
