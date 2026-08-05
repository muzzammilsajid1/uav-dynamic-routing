import json
import numpy as np
import math
from pathlib import Path
import sys
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from rl_v3.phase_c1_env import PhaseC1Env, PhaseC1EndpointGenerator
from rl_v3.phase_c0_env import _astar_cost
from tools.verification.action_mapping import AUTHORITATIVE_ACTION_MAPPING

class PhaseC1Oracle:
    def __init__(self, env=None):
        self.env = env
        
    def octile_distance(self, p1, p2):
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)

    def predict_from_pos(self, pos, goal, mask=None):
        best_action = None
        best_val = float('inf')
        
        for a in range(8):
            if mask is not None and not mask[a]:
                continue
            delta = AUTHORITATIVE_ACTION_MAPPING[a]["delta"]
            cost = AUTHORITATIVE_ACTION_MAPPING[a]["cost"]
            next_pos = (pos[0] + delta[0], pos[1] + delta[1])
            dist = self.octile_distance(next_pos, goal)
            val = cost + dist
            
            # Tie breaking by smallest action index implicitly
            if val < best_val - 1e-9:
                best_val = val
                best_action = a
        return best_action
        
    def predict(self, state, mask):
        env = self.env.unwrapped
        pos = env._v2.uav_pos
        goal = env._v2.goal_pos
        return self.predict_from_pos(pos, goal, mask)

def eval_all_pairs():
    with open(ROOT / "configs" / "rl_v3_phase_c1.json") as f:
        config_text = f.read()
        config = json.loads(config_text)
        
    config_hash = hashlib.sha256(config_text.encode('utf-8')).hexdigest()
    
    with open(Path(__file__).resolve(), "r") as f:
        source_hash = hashlib.sha256(f.read().encode('utf-8')).hexdigest()
        
    with open(ROOT / "tools" / "verification" / "action_mapping.py", "r") as f:
        action_hash = hashlib.sha256(f.read().encode('utf-8')).hexdigest()

    gen = PhaseC1EndpointGenerator(seed=42)
    env = PhaseC1Env(config, mode="train", generator=gen)
    oracle = PhaseC1Oracle(env)
    
    # Generate 50,400 pairs
    pairs = []
    for sx in range(15):
        for sy in range(15):
            for gx in range(15):
                for gy in range(15):
                    if sx == gx and sy == gy: continue
                    pairs.append(((sx, sy), (gx, gy)))
                    
    metrics = {
        "successes": 0,
        "collisions": 0,
        "timeouts": 0,
        "routes": len(pairs),
        "non_zero_gap_routes": 0,
        "max_absolute_gap": 0.0,
    }
    
    for start, goal in pairs:
        env.reset()
        env.unwrapped._start = start
        env.unwrapped._goal = goal
        env.unwrapped._v2.uav_pos = np.array(start, dtype=int)
        env.unwrapped._v2.goal_pos = np.array(goal, dtype=int)
        
        astar_cost = _astar_cost(start, goal)
        env.unwrapped._max_steps = max(10, int(math.ceil(astar_cost * float(config["training"]["episode_budget_astar_multiplier"]))))
        
        path_cost = 0.0
        done = False
        
        while not done:
            mask = env.action_masks()
            action = oracle.predict(None, mask)
            
            path_cost += AUTHORITATIVE_ACTION_MAPPING[action]["cost"]
            
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            
        if info.get("is_success"):
            metrics["successes"] += 1
            gap = abs(path_cost - astar_cost)
            metrics["max_absolute_gap"] = max(metrics["max_absolute_gap"], gap)
            if gap > 1e-6:
                metrics["non_zero_gap_routes"] += 1
        elif info.get("crashed"):
            metrics["collisions"] += 1
        else:
            metrics["timeouts"] += 1
            
    out_path = ROOT / "runs" / "rl_v3" / "phase_c1_diagnostic" / "oracle_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    result_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_hash": source_hash,
        "action_mapping_hash": action_hash,
        "config_hash": config_hash,
        "route_count": metrics["routes"],
        "aggregate_metrics": metrics,
        "max_absolute_path_cost_gap": metrics["max_absolute_gap"],
        "count_non_zero_gap_routes": metrics["non_zero_gap_routes"]
    }
    
    temp_path = out_path.with_suffix('.tmp')
    with open(temp_path, "w") as f:
        json.dump(result_data, f, indent=2)
    temp_path.replace(out_path)
    
    print(f"Oracle test completed. Gap > 0 routes: {metrics['non_zero_gap_routes']}")

if __name__ == "__main__":
    eval_all_pairs()
