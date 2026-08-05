import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from rl_v3.phase_c1_env import PhaseC1Env, PhaseC1EndpointGenerator
from tools.verification.oracle import PhaseC1Oracle
from rl_v3.phase_b_policy import PhaseBFeatureExtractor
from stable_baselines3.common.env_util import make_vec_env

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
    
    for _ in range(max_episodes):
        obs, info = env.reset()
        
        start = env.unwrapped._start
        goal = env.unwrapped._goal
        gen = env.unwrapped.generator
        from rl_v3.phase_c0_env import _astar_cost
        astar = _astar_cost(start, goal)
        if astar < gen.T1: dist_bin = "short"
        elif astar < gen.T2: dist_bin = "medium"
        else: dist_bin = "long"
        ori = gen.orientations.get((start, goal), "mixed")
        
        done = False
        
        while not done:
            mask = env.action_masks()
            action = oracle.predict(obs, mask)
            
            # Save the state
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
            
    return ExpressivityDataset(dataset)

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

def train_classifier(model, train_loader, val_loader, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
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
        "total": 0,
        "correct": 0,
        "illegal_predicted": 0,
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
                
                metrics["bin_total"][b] += 1
                if o in metrics["ori_total"]:
                    metrics["ori_total"][o] += 1
                else:
                    metrics["ori_total"]["mixed"] += 1
                    o = "mixed"
                
                if m[i][preds_raw[i]] == 0.0:
                    metrics["illegal_predicted"] += 1
                
                # Accuracy after masking (legal greedy)
                legal_logits = logits[i].clone()
                legal_logits[m[i] == 0] = -float('inf')
                pred_legal = torch.argmax(legal_logits)
                
                if pred_legal == a[i]:
                    metrics["correct"] += 1
                    metrics["bin_correct"][b] += 1
                    metrics["ori_correct"][o] += 1
                    
    return metrics

def rollout_classifier(model, env_config, gen, num_episodes):
    env = PhaseC1Env(env_config, mode="eval", generator=gen)
    model.eval()
    successes = 0
    
    with torch.no_grad():
        for _ in range(num_episodes):
            obs, info = env.reset()
            done = False
            while not done:
                mask = env.action_masks()
                
                # prepare obs for model
                g = torch.tensor(obs["global_map"], dtype=torch.float32).unsqueeze(0)
                l = torch.tensor(obs["local_map"], dtype=torch.float32).unsqueeze(0)
                s = torch.tensor(obs["scalars"], dtype=torch.float32).unsqueeze(0)
                m = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
                
                logits = model({"global_map": g, "local_map": l, "scalars": s})
                logits[m == 0] = -float('inf')
                action = torch.argmax(logits, dim=1).item()
                
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                
            if info.get("is_success"):
                successes += 1
                
    return successes / float(num_episodes)

def run():
    with open(ROOT / "configs" / "rl_v3_phase_c1.json") as f:
        config = json.load(f)
        
    print("Collecting datasets...")
    train_gen = PhaseC1EndpointGenerator(seed=42)
    train_env = PhaseC1Env(config, mode="train", generator=train_gen)
    train_oracle = PhaseC1Oracle(train_env)
    train_ds = collect_dataset(train_env, train_oracle, max_episodes=500)
    
    val_gen = PhaseC1EndpointGenerator(seed=999)
    val_env = PhaseC1Env(config, mode="eval", generator=val_gen)
    val_oracle = PhaseC1Oracle(val_env)
    val_ds = collect_dataset(val_env, val_oracle, max_episodes=120)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    eval_train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    
    print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")
    
    # E1
    e1 = E1Model(scalar_dim=4, num_actions=8)
    print("Training E1 (Scalar Only)...")
    train_classifier(e1, train_loader, val_loader, epochs=25)
    e1_train_metrics = evaluate_classifier(e1, eval_train_loader)
    e1_val_metrics = evaluate_classifier(e1, val_loader)
    e1_rollout_sr = rollout_classifier(e1, config, PhaseC1EndpointGenerator(seed=42), 120)
    
    # E2
    # Needs observation space for PhaseBFeatureExtractor
    from rl_v3.observations import observation_space
    obs_space = observation_space(11, 32)
    e2 = E2Model(obs_space, num_actions=8)
    print("Training E2 (Global-Local CNN)...")
    train_classifier(e2, train_loader, val_loader, epochs=25)
    e2_train_metrics = evaluate_classifier(e2, eval_train_loader)
    e2_val_metrics = evaluate_classifier(e2, val_loader)
    e2_rollout_sr = rollout_classifier(e2, config, PhaseC1EndpointGenerator(seed=42), 120)
    
    res = {
        "E1": {
            "train_acc": e1_train_metrics["correct"] / e1_train_metrics["total"],
            "val_acc": e1_val_metrics["correct"] / e1_val_metrics["total"],
            "val_illegal_rate": e1_val_metrics["illegal_predicted"] / e1_val_metrics["total"],
            "rollout_success": e1_rollout_sr,
            "val_by_bin": {b: (e1_val_metrics["bin_correct"][b] / e1_val_metrics["bin_total"][b] if e1_val_metrics["bin_total"][b] else 0) for b in ["short", "medium", "long"]},
            "val_by_ori": {o: (e1_val_metrics["ori_correct"][o] / e1_val_metrics["ori_total"][o] if e1_val_metrics["ori_total"][o] else 0) for o in ["vertical", "horizontal", "diagonal", "mixed"]}
        },
        "E2": {
            "train_acc": e2_train_metrics["correct"] / e2_train_metrics["total"],
            "val_acc": e2_val_metrics["correct"] / e2_val_metrics["total"],
            "val_illegal_rate": e2_val_metrics["illegal_predicted"] / e2_val_metrics["total"],
            "rollout_success": e2_rollout_sr,
            "val_by_bin": {b: (e2_val_metrics["bin_correct"][b] / e2_val_metrics["bin_total"][b] if e2_val_metrics["bin_total"][b] else 0) for b in ["short", "medium", "long"]},
            "val_by_ori": {o: (e2_val_metrics["ori_correct"][o] / e2_val_metrics["ori_total"][o] if e2_val_metrics["ori_total"][o] else 0) for o in ["vertical", "horizontal", "diagonal", "mixed"]}
        }
    }
    
    out_dir = ROOT / "runs" / "rl_v3" / "phase_c1_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "expressivity_results.json", "w") as f:
        json.dump(res, f, indent=2)
        
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    run()
