import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys
import hashlib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from rl_v3.phase_c1_env import PhaseC1Env, PhaseC1EndpointGenerator
from tools.verification.oracle import PhaseC1Oracle
from rl_v3.phase_b_policy import PhaseBFeatureExtractor
from tools.verification.action_mapping import AUTHORITATIVE_ACTION_MAPPING
from rl_v3.phase_c0_env import _astar_cost

class ExpressivityDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return (
            torch.tensor(item["global_map"], dtype=torch.float32),
            torch.tensor(item["local_map"], dtype=torch.float32),
            torch.tensor(item["scalars"], dtype=torch.float32),
            torch.tensor(item["mask"], dtype=torch.float32),
            torch.tensor(item["action"], dtype=torch.long),
            item["distance_bin"],
            item["orientation"]
        )

def collect_dataset(env, oracle, max_episodes):
    dataset = []
    pairs = set()
    action_counts = {a: 0 for a in range(8)}
    
    for _ in range(max_episodes):
        obs, info = env.reset()
        start = env.unwrapped._start
        goal = env.unwrapped._goal
        pairs.add((start, goal))
        
        gen = env.unwrapped.generator
        astar = _astar_cost(start, goal)
        if astar < gen.T1: dist_bin = "short"
        elif astar < gen.T2: dist_bin = "medium"
        else: dist_bin = "long"
        ori = gen.orientations.get((start, goal), "mixed")
        
        done = False
        while not done:
            mask = env.action_masks()
            action = oracle.predict(obs, mask)
            action_counts[action] += 1
            
            dataset.append({
                "global_map": obs["global_map"],
                "local_map": obs["local_map"],
                "scalars": obs["scalars"],
                "mask": mask,
                "action": action,
                "distance_bin": dist_bin,
                "orientation": ori
            })
            
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
            
    return ExpressivityDataset(dataset), list(pairs), action_counts

class E1Model(nn.Module):
    def __init__(self, scalar_dim, num_actions=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(scalar_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
    def forward(self, obs_dict):
        return self.net(obs_dict["scalars"])

class E2Model(nn.Module):
    def __init__(self, obs_space, num_actions=8):
        super().__init__()
        self.extractor = PhaseBFeatureExtractor(obs_space, features_dim=256)
        self.head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )
        
    def forward(self, obs_dict):
        features = self.extractor(obs_dict)
        return self.head(features)

def train_classifier(model, train_loader, val_loader, epochs=25, lr=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        model.train()
        for g, l, s, m, a, _, _ in train_loader:
            obs = {"global_map": g, "local_map": l, "scalars": s}
            logits = model(obs)
            loss = criterion(logits, a)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_classifier(model, data_loader):
    model.eval()
    metrics = {
        "total": 0, "correct": 0, "illegal_predicted": 0,
        "action_correct": {str(a): 0 for a in range(8)},
        "action_total": {str(a): 0 for a in range(8)},
        "ori_correct": {"vertical": 0, "horizontal": 0, "diagonal": 0, "mixed": 0},
        "ori_total": {"vertical": 0, "horizontal": 0, "diagonal": 0, "mixed": 0},
        "bin_correct": {"short": 0, "medium": 0, "long": 0},
        "bin_total": {"short": 0, "medium": 0, "long": 0}
    }
    
    with torch.no_grad():
        for g, l, s, m, a, bins, oris in data_loader:
            obs = {"global_map": g, "local_map": l, "scalars": s}
            logits = model(obs)
            preds_raw = torch.argmax(logits, dim=1)
            
            for i in range(len(preds_raw)):
                metrics["total"] += 1
                b = bins[i]
                o = oris[i]
                act = str(a[i].item())
                
                metrics["bin_total"][b] += 1
                if o in metrics["ori_total"]: metrics["ori_total"][o] += 1
                else: 
                    metrics["ori_total"]["mixed"] += 1
                    o = "mixed"
                metrics["action_total"][act] += 1
                
                if m[i][preds_raw[i]] == 0.0:
                    metrics["illegal_predicted"] += 1
                
                legal_logits = logits[i].clone()
                legal_logits[m[i] == 0] = -float('inf')
                pred_legal = torch.argmax(legal_logits)
                
                if pred_legal == a[i]:
                    metrics["correct"] += 1
                    metrics["bin_correct"][b] += 1
                    metrics["ori_correct"][o] += 1
                    metrics["action_correct"][act] += 1
                    
    # compute percentages
    ret = {
        "accuracy": metrics["correct"] / metrics["total"] if metrics["total"] > 0 else 0,
        "illegal_rate": metrics["illegal_predicted"] / metrics["total"] if metrics["total"] > 0 else 0,
        "accuracy_by_bin": {k: v / metrics["bin_total"][k] if metrics["bin_total"][k] > 0 else 0 for k, v in metrics["bin_correct"].items()},
        "accuracy_by_ori": {k: v / metrics["ori_total"][k] if metrics["ori_total"][k] > 0 else 0 for k, v in metrics["ori_correct"].items()},
        "accuracy_by_action": {k: v / metrics["action_total"][k] if metrics["action_total"][k] > 0 else 0 for k, v in metrics["action_correct"].items()},
    }
    return ret

def rollout_classifier(model, env_config, gen, num_episodes):
    env = PhaseC1Env(env_config, mode="eval", generator=gen)
    model.eval()
    successes = 0
    collisions = 0
    timeouts = 0
    loops = 0
    total_cost = 0.0
    astar_total = 0.0
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, info = env.reset()
            astar_total += _astar_cost(env.unwrapped._start, env.unwrapped._goal)
            done = False
            route_cost = 0.0
            
            while not done:
                mask = env.action_masks()
                g = torch.tensor(obs["global_map"], dtype=torch.float32).unsqueeze(0)
                l = torch.tensor(obs["local_map"], dtype=torch.float32).unsqueeze(0)
                s = torch.tensor(obs["scalars"], dtype=torch.float32).unsqueeze(0)
                m = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                
                logits = model({"global_map": g, "local_map": l, "scalars": s})
                logits[m == 0] = -float('inf')
                action = torch.argmax(logits, dim=1).item()
                
                route_cost += AUTHORITATIVE_ACTION_MAPPING[action]["cost"]
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                
            if info.get("is_success"):
                successes += 1
                total_cost += route_cost
            elif info.get("crashed"):
                collisions += 1
            else:
                if info.get("is_loop"):
                    loops += 1
                else:
                    timeouts += 1
                
    return {
        "success_rate": successes / float(num_episodes),
        "collisions": collisions,
        "timeouts": timeouts,
        "loops": loops,
        "weighted_path_cost_ratio": total_cost / astar_total if successes == num_episodes else None
    }

def run():
    with open(ROOT / "configs" / "rl_v3_phase_c1.json") as f:
        config = json.load(f)
        
    print("Collecting datasets...")
    # Train
    train_gen = PhaseC1EndpointGenerator(seed=42)
    train_env = PhaseC1Env(config, mode="train", generator=train_gen)
    train_oracle = PhaseC1Oracle(train_env)
    train_ds, train_pairs, train_act_counts = collect_dataset(train_env, train_oracle, max_episodes=500)
    
    # Eval
    val_gen = PhaseC1EndpointGenerator(seed=999)
    val_env = PhaseC1Env(config, mode="eval", generator=val_gen)
    val_oracle = PhaseC1Oracle(val_env)
    val_ds, val_pairs, val_act_counts = collect_dataset(val_env, val_oracle, max_episodes=120)
    
    # Hash check
    ds_hash = hashlib.sha256((str(train_pairs) + str(val_pairs)).encode()).hexdigest()
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    eval_train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    
    torch.manual_seed(1337)
    
    # E1
    e1 = E1Model(scalar_dim=4, num_actions=8)
    train_classifier(e1, train_loader, val_loader, epochs=25, lr=1e-3)
    e1_train = evaluate_classifier(e1, eval_train_loader)
    e1_val = evaluate_classifier(e1, val_loader)
    e1_rollout = rollout_classifier(e1, config, PhaseC1EndpointGenerator(seed=42), 120)
    
    # E2
    from rl_v3.observations import observation_space
    obs_space = observation_space(11, 32)
    e2 = E2Model(obs_space, num_actions=8)
    train_classifier(e2, train_loader, val_loader, epochs=25, lr=1e-3)
    e2_train = evaluate_classifier(e2, eval_train_loader)
    e2_val = evaluate_classifier(e2, val_loader)
    e2_rollout = rollout_classifier(e2, config, PhaseC1EndpointGenerator(seed=42), 120)
    
    report = {
        "metadata": {
            "dataset_generation_seeds": {"train": 42, "eval": 999},
            "dataset_hash": ds_hash,
            "train_pairs": train_pairs,
            "eval_pairs": val_pairs,
            "split_rule": "Deterministic Phase C1 Validation Manifest holdout",
            "model_seed": 1337,
            "network_specs": {
                "E1": "MLP(4 -> 64 -> 64 -> 8)",
                "E2": "PhaseBFeatureExtractor(256) -> MLP(64 -> 8)"
            },
            "optimizer": "Adam, lr=1e-3, batch_size=64",
            "epochs": 25,
            "class_distribution": {
                "train": train_act_counts,
                "eval": val_act_counts
            }
        },
        "E1": {
            "training_metrics": e1_train,
            "held_out_metrics": e1_val,
            "rollout_metrics": e1_rollout
        },
        "E2": {
            "training_metrics": e2_train,
            "held_out_metrics": e2_val,
            "rollout_metrics": e2_rollout
        }
    }
    
    out_dir = ROOT / "runs" / "rl_v3" / "phase_c1_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "expressivity_results.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    run()
