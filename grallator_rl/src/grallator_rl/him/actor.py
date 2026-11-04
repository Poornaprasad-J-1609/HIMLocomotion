import math

import torch
import torch.nn as nn

from torch.distributions import Normal


class HIMActor(nn.Module):

    def __init__(self, estimator, init_std=0.3):
        super().__init__()

        if init_std <= 0:
            raise ValueError("init_std must be positive")

        self.estimator = estimator

        # 45 current obs + 3 velocity + 16 latent = 64
        self.actor = nn.Sequential(
            nn.Linear(64, 512),
            nn.ELU(),

            nn.Linear(512, 256),
            nn.ELU(),

            nn.Linear(256, 128),
            nn.ELU(),

            nn.Linear(128, 12),
        )

        # Trainable Gaussian exploration parameters
        self.log_std = nn.Parameter(
            torch.full((12,), math.log(init_std))
        )

    def build_actor_input(self, obs_history):

        if (
            obs_history.ndim != 2
            or obs_history.shape[1] != 270
        ):
            raise ValueError(
                f"Expected history [N,270], "
                f"got {tuple(obs_history.shape)}"
            )

        # HIM extracts estimated velocity and latent.
        # Gradients are blocked during the PPO actor update.
        with torch.no_grad():
            estimated_velocity, latent = (
                self.estimator(obs_history)
            )

        # HIMHistory puts newest observation first
        current_obs = obs_history[:, :45]

        actor_input = torch.cat(
            (
                current_obs,
                estimated_velocity,
                latent,
            ),
            dim=-1,
        )

        assert actor_input.shape[1] == 64

        return actor_input

    def forward(self, obs_history):

        actor_input = self.build_actor_input(obs_history)

        # The 12 Gaussian means
        return self.actor(actor_input)

    def action_distribution(self, obs_history):

        mean = self.forward(obs_history)

        std = self.log_std.exp().expand_as(mean)

        return Normal(mean, std)

    def act_inference(self, obs_history):

        # Deterministic actions for evaluation
        return self.forward(obs_history)