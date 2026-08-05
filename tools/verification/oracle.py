import json
import numpy as np
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from rl_v3.phase_c1_env import PhaseC1Env, PhaseC1EndpointGenerator
from rl_v3.phase_c0_env import _astar_cost

class PhaseC1Oracle:
    def __init__(self, env):
        self.env = env
        
    def octile_distance(self, p1, p2):
        dx = abs(p1[0] - p2[0])
        dy = abs(p1[1] - p2[1])
        return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)

    def predict(self, state, mask):
        # We need to simulate the next state for each legal action.
        # But wait, PhaseC1Env is a gym environment. We can't trivially simulate without stepping and unwinding.
        # However, the dynamics are perfectly deterministic and known:
        # action 0-7 are 8-way movement, action 8 is hover.
        # The env's current position is self.env.unwrapped._pos
        # The goal is self.env.unwrapped._goal
        
        env = self.env.unwrapped
        pos = env._v2.uav_pos
        goal = env._v2.goal_pos
        
        best_action = None
        best_dist = float('inf')
        
        # Action space mapping from UAVRoutingEnv.ACTION_DELTAS:
        # [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        action_deltas = {
            0: (-1, 0),
            1: (1, 0),
            2: (0, -1),
            3: (0, 1),
            4: (-1, -1),
            5: (-1, 1),
            6: (1, -1),
            7: (1, 1)
        }
        
        for a in range(8):
            if mask[a]:
                dr, dc = action_deltas[a]
                next_pos = (pos[0] + dr, pos[1] + dc)
                dist = self.octile_distance(next_pos, goal)
                if dist < best_dist:
                    best_dist = dist
                    best_action = a
                    
        return best_action

def evaluate_oracle(env, num_episodes):
    oracle = PhaseC1Oracle(env)
    
    metrics = {
        "successes": 0,
        "collisions": 0,
        "timeouts": 0,
        "realized_cost_sum": 0.0,
        "astar_cost_sum": 0.0,
        "path_cost_gap_sum": 0.0,
        "route_steps_sum": 0,
        "orientation": {},
        "distance_bin": {"short": 0, "medium": 0, "long": 0}
    }
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        realized_cost = 0.0
        
        start = env.unwrapped._start
        goal = env.unwrapped._goal
        astar = _astar_cost(start, goal)
        metrics["astar_cost_sum"] += astar
        
        gen = env.unwrapped.generator
        if astar < gen.T1: dist_bin = "short"
        elif astar < gen.T2: dist_bin = "medium"
        else: dist_bin = "long"
        
        ori = gen.orientations.get((start, goal), "mixed")
        if ori not in metrics["orientation"]:
            metrics["orientation"][ori] = 0
            
        while not done:
            mask = env.action_masks()
            action = oracle.predict(obs, mask)
            
            # For cost: orthogonal = 1, diagonal = sqrt(2), hover = 0.2
            if action == 8: step_cost = 0.2
            elif action in [0,2,4,6]: step_cost = 1.0
            else: step_cost = np.sqrt(2)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            realized_cost += step_cost
            
        if info.get("is_success"):
            metrics["successes"] += 1
            metrics["realized_cost_sum"] += realized_cost
            metrics["path_cost_gap_sum"] += (realized_cost - astar)
            metrics["route_steps_sum"] += steps
            metrics["orientation"][ori] += 1
            metrics["distance_bin"][dist_bin] += 1
        elif info.get("is_collision"):
            metrics["collisions"] += 1
        elif info.get("is_timeout"):
            metrics["timeouts"] += 1
            
    return metrics

def run_oracle_sanity_test():
    with open(ROOT / "configs" / "rl_v3_phase_c1.json") as f:
        config = json.load(f)
        
    print("Running Oracle on Validation Set (120 pairs)...")
    val_gen = PhaseC1EndpointGenerator(seed=42)
    val_env = PhaseC1Env(config, mode="eval", generator=val_gen)
    
    val_metrics = evaluate_oracle(val_env, 120)
    print("Validation Oracle Metrics:", val_metrics)
    
    print("Running Oracle on Training Set (1000 pairs)...")
    train_gen = PhaseC1EndpointGenerator(seed=42)
    train_env = PhaseC1Env(config, mode="train", generator=train_gen)
    train_metrics = evaluate_oracle(train_env, 1000)
    print("Training Oracle Metrics:", train_metrics)
    
    report = {
        "validation_120": val_metrics,
        "training_1000": train_metrics
    }
    
    out_dir = ROOT / "runs" / "rl_v3" / "phase_c1_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "oracle_results.json", "w") as f:
        json.dump(report, f, indent=2)
        
if __name__ == "__main__":
    run_oracle_sanity_test()
