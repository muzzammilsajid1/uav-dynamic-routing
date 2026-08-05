import json
import math
from pathlib import Path
import numpy as np

from rl_v3.phase_c1_env import PhaseC1Env, PhaseC1EndpointGenerator
from rl_v3.phase_c0_env import _astar_cost
from rl_v3.run_phase_c1 import load_config

def run_verification():
    config = load_config()
    out_dir = Path("runs/rl_v3/phase_c1_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    mismatches = 0
    results = []
    
    # Train
    g_train = PhaseC1EndpointGenerator(seed=42)
    env_train = PhaseC1Env(config, mode="train", generator=g_train)
    
    for i in range(100):
        obs, info = env_train.reset()
        r = verify_single_reset(env_train, obs, info, config)
        r["mode"] = "train"
        r["index"] = i
        results.append(r)
        if not r["passed"]: mismatches += 1
        
    env_train.close()
        
    # Val
    g_val = PhaseC1EndpointGenerator(seed=42)
    env_val = PhaseC1Env(config, mode="val", generator=g_val)
    
    for i in range(120):
        obs, info = env_val.reset()
        r = verify_single_reset(env_val, obs, info, config)
        r["mode"] = "val"
        r["index"] = i
        results.append(r)
        if not r["passed"]: mismatches += 1
        
    env_val.close()
    
    report = {
        "mismatches": mismatches,
        "details": results
    }
    
    with open(out_dir / "endpoint_application_verification.json", "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Mismatches: {mismatches}")

def verify_single_reset(env, obs, info, config):
    passed = True
    errors = []
    
    sampled_start = env._start
    sampled_goal = env._goal
    
    uav_pos = tuple(env._v2.uav_pos)
    goal_pos = tuple(env._v2.goal_pos)
    
    if sampled_start != uav_pos:
        passed = False
        errors.append(f"sampled_start {sampled_start} != uav_pos {uav_pos}")
        
    if info["start"] != uav_pos:
        passed = False
        errors.append(f"info['start'] {info['start']} != uav_pos {uav_pos}")
        
    if sampled_goal != goal_pos:
        passed = False
        errors.append(f"sampled_goal {sampled_goal} != goal_pos {goal_pos}")
        
    if info["goal"] != goal_pos:
        passed = False
        errors.append(f"info['goal'] {info['goal']} != goal_pos {goal_pos}")
        
    uav_r, uav_c = uav_pos
    goal_r, goal_c = goal_pos
    
    s_r = (goal_r - uav_r) / 14.0
    s_c = (goal_c - uav_c) / 14.0
    
    if not np.isclose(obs["scalars"][0], s_r):
        passed = False
        errors.append(f"scalar[0] {obs['scalars'][0]} != {s_r}")
        
    if not np.isclose(obs["scalars"][1], s_c):
        passed = False
        errors.append(f"scalar[1] {obs['scalars'][1]} != {s_c}")
        
    cost = _astar_cost(sampled_start, sampled_goal)
    if info["astar_cost"] != cost:
        passed = False
        errors.append(f"info['astar_cost'] {info['astar_cost']} != {cost}")
        
    budget = max(int(config["training"]["minimum_episode_budget"]), int(math.ceil(cost * float(config["training"]["episode_budget_astar_multiplier"]))))
    
    if env._max_steps != budget:
        passed = False
        errors.append(f"env._max_steps {env._max_steps} != budget {budget}")
        
    if info["budget"] != budget:
        passed = False
        errors.append(f"info['budget'] {info['budget']} != budget {budget}")
        
    return {
        "passed": passed,
        "errors": errors
    }

if __name__ == "__main__":
    run_verification()
