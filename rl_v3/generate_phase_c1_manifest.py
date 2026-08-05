import json
import hashlib
import os
import math
from pathlib import Path
import numpy as np

def _astar_cost(start, goal):
    dr = abs(goal[0] - start[0])
    dc = abs(goal[1] - start[1])
    diag = min(dr, dc)
    straight = max(dr, dc) - diag
    return diag * math.sqrt(2.0) + straight

def generate_manifest():
    grid_size = 15
    seed = 42
    rng = np.random.RandomState(seed)
    
    T1 = 6.2
    T2 = 10.1
    
    def get_region(p):
        return (p[0] // 5) * 3 + (p[1] // 5)
        
    def get_orientation(p0, p1):
        dx = abs(p1[0] - p0[0]) # row diff
        dy = abs(p1[1] - p0[1]) # col diff
        
        if dx > 0 and dy <= dx * 0.5:
            return "vertical"
        elif dy > 0 and dx <= dy * 0.5:
            return "horizontal"
        elif abs(dx - dy) <= 1:
            return "diagonal"
        else:
            return "mixed"

    all_pairs = []
    for sx in range(grid_size):
        for sy in range(grid_size):
            for gx in range(grid_size):
                for gy in range(grid_size):
                    if sx == gx and sy == gy: continue
                    all_pairs.append(((sx, sy), (gx, gy)))
                    
    bins = {"short": [], "medium": [], "long": []}
    for p in all_pairs:
        cost = _astar_cost(p[0], p[1])
        if cost < T1:
            b = "short"
        elif cost < T2:
            b = "medium"
        else:
            b = "long"
        
        bins[b].append({
            "pair": p,
            "cost": cost,
            "orientation": get_orientation(p[0], p[1]),
            "start_region": get_region(p[0]),
            "goal_region": get_region(p[1])
        })
        
    manifest = []
    excluded_set = set()
    
    multiplier = 5.0
    minimum_budget = 30
    
    for b in ["short", "medium", "long"]:
        candidates = bins[b]
        rng.shuffle(candidates)
        
        selected_for_bin = []
        
        ori_buckets = {"vertical": [], "horizontal": [], "diagonal": [], "mixed": []}
        for c in candidates:
            ori_buckets[c["orientation"]].append(c)
            
        for ori in ["vertical", "horizontal", "diagonal", "mixed"]:
            needed = 10
            for c in ori_buckets[ori]:
                if len(selected_for_bin) >= 40:
                    break
                if needed <= 0:
                    break
                    
                p = c["pair"]
                rev = (p[1], p[0])
                if p in excluded_set or rev in excluded_set:
                    continue
                
                needed -= 1
                selected_for_bin.append(c)
                excluded_set.add(p)
                excluded_set.add(rev)
                
        if len(selected_for_bin) < 40:
            for c in candidates:
                if len(selected_for_bin) >= 40:
                    break
                p = c["pair"]
                rev = (p[1], p[0])
                if p not in excluded_set and rev not in excluded_set:
                    selected_for_bin.append(c)
                    excluded_set.add(p)
                    excluded_set.add(rev)
                    
        for c in selected_for_bin:
            manifest.append(c)
            
    final_manifest = []
    for idx, c in enumerate(manifest):
        p = c["pair"]
        cost = c["cost"]
        budget = max(minimum_budget, int(math.ceil(cost * multiplier)))
        final_manifest.append({
            "scenario_id": f"val_{idx:03d}",
            "start": p[0],
            "goal": p[1],
            "reverse_pair_exclusion_key": f"{p[1][0]}_{p[1][1]}_to_{p[0][0]}_{p[0][1]}",
            "weighted_astar_cost": cost,
            "distance_bin": "short" if cost < T1 else ("medium" if cost < T2 else "long"),
            "route_orientation": c["orientation"],
            "start_region": c["start_region"],
            "goal_region": c["goal_region"],
            "episode_budget": budget
        })
        
    out_dir = Path("evaluation/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rl_v3_phase_c1_validation.json"
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2, sort_keys=True)
        
    h = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"Manifest written to {out_path}")
    print(f"Hash: {h}")
    print(f"Total pairs: {len(final_manifest)}")
    
if __name__ == "__main__":
    generate_manifest()
