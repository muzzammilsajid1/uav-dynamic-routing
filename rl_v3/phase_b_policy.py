"""Moderate multi-branch feature extractor for Phase B Maskable PPO."""
from __future__ import annotations

import time

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PhaseBFeatureExtractor(BaseFeaturesExtractor):
    """Encode local/global spatial maps separately, then concatenate scalars."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.local_encoder = _cnn_branch(observation_space["local_map"].shape, 96)
        self.has_global = "global_map" in observation_space.spaces
        if self.has_global:
            self.global_encoder = _cnn_branch(observation_space["global_map"].shape, 128)
        self.scalar_encoder = torch.nn.Sequential(
            torch.nn.Linear(4, 32), torch.nn.Tanh(), torch.nn.Linear(32, 32), torch.nn.Tanh()
        )
        concat = 96 + 32 + (128 if self.has_global else 0)
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(concat, features_dim), torch.nn.ReLU()
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        pieces = [self.local_encoder(observations["local_map"])]
        if self.has_global:
            pieces.append(self.global_encoder(observations["global_map"]))
        pieces.append(self.scalar_encoder(observations["scalars"]))
        return self.fusion(torch.cat(pieces, dim=1))


def _cnn_branch(shape: tuple[int, ...], output: int) -> torch.nn.Sequential:
    channels, height, width = shape
    conv = torch.nn.Sequential(
        torch.nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
    )
    with torch.no_grad():
        flat = int(conv(torch.zeros(1, channels, height, width)).shape[1])
    return torch.nn.Sequential(conv, torch.nn.Linear(flat, output), torch.nn.ReLU())


def model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def inference_latency_ms(model, observation: dict, action_masks: np.ndarray, repeats: int = 200) -> float:
    for _ in range(10):
        model.predict(observation, deterministic=True, action_masks=action_masks)
    started = time.perf_counter()
    for _ in range(repeats):
        model.predict(observation, deterministic=True, action_masks=action_masks)
    return 1000.0 * (time.perf_counter() - started) / repeats
