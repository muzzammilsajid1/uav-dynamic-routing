import json
import random
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from rl_v3.phase_c1_env import PhaseC1EndpointGenerator

def generate_ladder_manifests():
    val_path = ROOT / "evaluation" / "manifests" / "rl_v3_phase_c1_validation.json"
    with open(val_path) as f:
        val_data = json.load(f)
        val_pairs = [(tuple(d["start"]), tuple(d["goal"])) for d in val_data]
        
    val_set = set(val_pairs)
    # Also exclude reverse routes
    val_set.update([(g, s) for (s, g) in val_pairs])
    
    # We want perfectly nested manifests: D1=4, D2=16, D3=64, D4=256
    # We can just sample from all valid pairs
    all_valid = []
    
    gen = PhaseC1EndpointGenerator(seed=0)
    for s in range(15):
        for c1 in range(15):
            for g1 in range(15):
                for c2 in range(15):
                    start = (s, c1)
                    goal = (g1, c2)
                    if start == goal: continue
                    if (start, goal) not in val_set:
                        all_valid.append((start, goal))
                        
    # Try to balance them roughly
    random.seed(42)
    random.shuffle(all_valid)
    
    def select_n(pool, n, exclude_set):
        selected = []
        for p in pool:
            if len(selected) == n: break
            if p not in exclude_set:
                selected.append(p)
                exclude_set.add(p)
        return selected

    used = set()
    d1 = select_n(all_valid, 4, used)
    d2 = d1 + select_n(all_valid, 12, used)
    d3 = d2 + select_n(all_valid, 48, used)
    d4 = d3 + select_n(all_valid, 192, used)
    
    out_dir = ROOT / "evaluation" / "manifests"
    
    for name, data in [("D1", d1), ("D2", d2), ("D3", d3), ("D4", d4)]:
        formatted = [{"start": list(s), "goal": list(g)} for s, g in data]
        with open(out_dir / f"rl_v3_ladder_{name}.json", "w") as f:
            json.dump(formatted, f, indent=2)
            
    print("Ladder manifests generated.")
    
if __name__ == "__main__":
    generate_ladder_manifests()
