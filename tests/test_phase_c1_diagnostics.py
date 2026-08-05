import pytest
import json
import torch
from pathlib import Path
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_v3.phase_c1_env import PhaseC1Env, PhaseC1EndpointGenerator
from tools.verification.oracle import PhaseC1Oracle
from tools.verification.expressivity import collect_dataset, E1Model, train_classifier
from rl_v3.phase_b_policy import PhaseBFeatureExtractor

ROOT = Path(__file__).resolve().parents[1]

def make_test_config():
    with open(ROOT / "configs" / "rl_v3_phase_c1.json") as f:
        return json.load(f)

def test_oracle_miniature():
    """Run Oracle Audit on miniature 10-pair pseudo-manifest."""
    config = make_test_config()
    gen = PhaseC1EndpointGenerator(seed=123)
    # create dummy 10-pair validation manifest
    gen.val_manifest = [((0, 0), (x, y)) for x, y in zip(range(1, 11), range(1, 11))]
    env = PhaseC1Env(config, mode="eval", generator=gen)
    oracle = PhaseC1Oracle(env)
    
    successes = 0
    for _ in range(10):
        obs, info = env.reset()
        done = False
        while not done:
            mask = env.action_masks()
            action = oracle.predict(obs, mask)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
        if info.get("is_success"):
            successes += 1
            
    assert successes == 10, "Oracle failed on miniature dataset."

def test_expressivity_audit():
    """Run Expressivity Audit on a 100-pair dataset."""
    config = make_test_config()
    gen = PhaseC1EndpointGenerator(seed=456)
    env = PhaseC1Env(config, mode="train", generator=gen)
    oracle = PhaseC1Oracle(env)
    
    dataset = collect_dataset(env, oracle, max_episodes=100)
    assert len(dataset) > 50, "Dataset too small"
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    
    e1 = E1Model(scalar_dim=4, num_actions=8)
    train_classifier(e1, loader, loader, epochs=1) # Just 1 epoch to ensure it runs fast
    
    # Assert model predicts correctly at a rate better than random chance
    # after 1 epoch (random is 1/8 = 12.5%, should easily exceed 30%)
    model = e1.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for g, l, s, m, a, bins, oris in loader:
            obs = {"scalars": s}
            logits = model(obs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == a).sum().item()
            total += len(a)
    
    accuracy = correct / total
    assert accuracy > 0.15, f"E1 model failed to learn anything (acc: {accuracy})"

def test_cardinality_ladder_failure():
    """Run D1 only (4 pairs) for 1000 interactions, expect failure to master."""
    config = make_test_config()
    
    def _init():
        gen = PhaseC1EndpointGenerator(seed=789)
        gen.train_bins = {
            "short": [((0, 0), (2, 2)), ((14, 14), (12, 12))],
            "medium": [((0, 14), (2, 12))],
            "long": [((14, 0), (12, 2))]
        }
        for b in gen.train_bins: gen.train_bins[b].sort()
        gen.orientations = {p: "mixed" for b in gen.train_bins.values() for p in b}
        return PhaseC1Env(config, mode="train", generator=gen)
        
    train_env = DummyVecEnv([_init])
    
    policy_kwargs = {
        "features_extractor_class": PhaseBFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128}, # Smaller for speed
        "net_arch": [64, 64],
    }
    
    model = MaskablePPO(
        "MultiInputPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=2,
        policy_kwargs=policy_kwargs,
        seed=42,
    )
    
    model.learn(total_timesteps=1000)
    
    # Evaluate D1 training success
    successes = 0
    env = _init()
    for _ in range(4):
        obs, info = env.reset()
        done = False
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
        if info.get("is_success"):
            successes += 1
            
    sr = successes / 4.0
    # The sparse reward should fail to master it in just 1000 steps
    assert sr < 0.9, f"Ladder learned too quickly? SR={sr}"
