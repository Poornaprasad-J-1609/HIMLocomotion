"""HIM RSL-RL actor: 270D history + 1D target -> 65D policy input.

No architecture change from the user's working 271D actor. Included to prevent
mixing incompatible 270D and 271D versions.
"""
from pathlib import Path

import torch
from rsl_rl.models import MLPModel

from grallator_rl.him.estimator import HIMEstimator


class HIMRslActorModel(MLPModel):
    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=False,
        distribution_cfg=None,
        estimator_checkpoint_path=None,
    ):
        if obs_set != "actor":
            raise ValueError("HIMRslActorModel is for the actor only")
        if obs_normalization:
            raise ValueError("Use the pretrained HIM observation convention")
        if distribution_cfg is None:
            raise ValueError("A Gaussian distribution is required")

        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )

        if self.obs_groups != ["policy"] or self.obs_dim != 271:
            raise ValueError("Actor requires 270D history + 1D height command")
        if output_dim != 12:
            raise ValueError("Grallator requires 12 actions")

        self.estimator = HIMEstimator(
            obs_dim=45,
            history_steps=6,
            latent_dim=16,
        )

        if estimator_checkpoint_path is not None:
            path = Path(estimator_checkpoint_path).expanduser()
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            self.estimator.load_state_dict(checkpoint["estimator"], strict=True)
            print("Loaded trained HIM estimator:", path)

    def _get_latent_dim(self):
        # Current proprioception + estimated velocity + HIM latent + height
        return 45 + 3 + 16 + 1  # 65, NOT 64

    def get_latent(self, obs, masks=None, hidden_state=None):
        policy_obs = obs["policy"]
        if policy_obs.ndim != 2 or policy_obs.shape[1] != 271:
            raise ValueError(f"Expected [N, 271], got {tuple(policy_obs.shape)}")

        history = policy_obs[:, :270]
        height_command = policy_obs[:, 270:271]

        # The HIM estimator is optimized separately by HIMTrainer.
        with torch.no_grad():
            velocity, latent = self.estimator.encode(history)

        current_obs = history[:, :45]
        actor_input = torch.cat(
            (current_obs, velocity, latent, height_command), dim=-1
        )
        assert actor_input.shape[1] == 65
        return actor_input
