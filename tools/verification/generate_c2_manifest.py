import json
import numpy as np
import hashlib
from pathlib import Path
import sys

def octile_distance(s, g):
    dx = abs(s[0] - g[0])
    dy = abs(s[1] - g[1])
    return max(dx, dy) + (np.sqrt(2) - 1) * min(dx, dy)

def get_bin(s, g, size):
    cost = octile_distance(s, g)
    max_d = size * np.sqrt(2)
    if cost < max_d / 3.0: return "short"
    elif cost < 2.0 * max_d / 3.0: return "medium"
    return "long"

def get_orientation(s, g):
    dx = abs(g[0] - s[0])
    dy = abs(g[1] - s[1])
    if dx > 0 and dy <= dx * 0.5: return "vertical"
    elif dy > 0 and dx <= dy * 0.5: return "horizontal"
    elif abs(dx - dy) <= 1: return "diagonal"
    return "mixed"

def generate_manifests():
    sizes = [15, 30, 50, 100]
    val_manifest = []
    train_pool = {str(sz): {"short": [], "medium": [], "long": []} for sz in sizes}
    
    rng = np.random.RandomState(42)
    
    # 60 val routes per size (20 short, 20 med, 20 long)
    val_set = set()
    for sz in sizes:
        for b in ["short", "medium", "long"]:
            count = 0
            while count < 20:
                sx, sy = rng.randint(0, sz, size=2)
                gx, gy = rng.randint(0, sz, size=2)
                if sx == gx and sy == gy: continue
                p = ((sx, sy), (gx, gy))
                rp = ((gx, gy), (sx, sy))
                if p in val_set or rp in val_set: continue
                if get_bin(p[0], p[1], sz) == b:
                    val_set.add(p)
                    val_set.add(rp)
                    val_manifest.append({
                        "grid_size": sz,
                        "distance_bin": b,
                        "orientation": get_orientation(p[0], p[1]),
                        "start": [int(sx), int(sy)],
                        "goal": [int(gx), int(gy)]
                    })
                    count += 1

    # Train pool: 5000 per bin per size
    for sz in sizes:
        for b in ["short", "medium", "long"]:
            count = 0
            while count < 5000:
                sx, sy = rng.randint(0, sz, size=2)
                gx, gy = rng.randint(0, sz, size=2)
                if sx == gx and sy == gy: continue
                p = ((sx, sy), (gx, gy))
                rp = ((gx, gy), (sx, sy))
                if p in val_set or rp in val_set: continue
                
                if get_bin(p[0], p[1], sz) == b:
                    train_pool[str(sz)][b].append([
                        [int(sx), int(sy)], [int(gx), int(gy)]
                    ])
                    count += 1
                    
    out_dir = Path("evaluation/manifests")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    val_path = out_dir / "rl_v3_phase_c2_validation.json"
    with open(val_path, "w") as f:
        json.dump(val_manifest, f, indent=2)
        
    train_path = out_dir / "rl_v3_phase_c2_train_generator.json"
    with open(train_path, "w") as f:
        json.dump(train_pool, f)
        
    print(f"Val SHA256: {hashlib.sha256(val_path.read_bytes()).hexdigest()}")
    print(f"Train SHA256: {hashlib.sha256(train_path.read_bytes()).hexdigest()}")

if __name__ == "__main__":
    generate_manifests()
