import json
import hashlib
from collections import Counter
from pathlib import Path
from rl_v3.phase_c0_env import _astar_cost

def verify_manifest():
    p = Path("evaluation/manifests/rl_v3_phase_c1_validation.json")
    b = p.read_bytes()
    h = hashlib.sha256(b).hexdigest()
    
    data = json.loads(b.decode('utf-8'))
    
    print(f"Path: {p}")
    print(f"Hash: {h}")
    print(f"Count: {len(data)}")
    
    bins = Counter([d["distance_bin"] for d in data])
    print(f"Bins: short={bins['short']}, medium={bins['medium']}, long={bins['long']}")
    
    ori = {"short": Counter(), "medium": Counter(), "long": Counter()}
    start_regions = {"short": Counter(), "medium": Counter(), "long": Counter()}
    goal_regions = {"short": Counter(), "medium": Counter(), "long": Counter()}
    
    invalid = 0
    unsolvable = 0
    pairs = []
    
    for d in data:
        b = d["distance_bin"]
        ori[b][d["route_orientation"]] += 1
        start_regions[b][d["start_region"]] += 1
        goal_regions[b][d["goal_region"]] += 1
        
        start = tuple(d["start"])
        goal = tuple(d["goal"])
        pairs.append((start, goal))
        
        if start == goal: invalid += 1
        if start[0] < 0 or start[0] >= 15 or start[1] < 0 or start[1] >= 15: invalid += 1
        if goal[0] < 0 or goal[0] >= 15 or goal[1] < 0 or goal[1] >= 15: invalid += 1
        
    print("Orientations:")
    for b_name in ["short", "medium", "long"]:
        print(f"  {b_name}: {dict(ori[b_name])}")
        
    print(f"Start regions: {dict(start_regions)}")
    print(f"Goal regions: {dict(goal_regions)}")
    
    duplicates = len(pairs) - len(set(pairs))
    print(f"Duplicates: {duplicates}")
    
    reverse_dups = 0
    pair_set = set(pairs)
    for p in pairs:
        if (p[1], p[0]) in pair_set:
            reverse_dups += 1
    
    print(f"Reverse pairs: {reverse_dups // 2}")
    print(f"Invalid pairs: {invalid}")
    print(f"Unsolvable pairs: {unsolvable}")

if __name__ == "__main__":
    verify_manifest()
