"""Fixed-height standing adapter: 271D actor / 92D critic.

The 45D original observation is unmodified (no height hidden in yaw slot).
The height command is appended separately *once*, outside the 270D history.
"""
from collections.abc import Sequence

import torch
from isaaclab.utils import configclass

from grallator_rl.him.history import HIMHistory
from grallator_rl.him.critic import CRITIC_OBS_DIM, build_critic_observation
from grallator_rl.tasks.standing_direct.config.grallator.env import StandingEnv
from grallator_rl.tasks.standing_direct.config.grallator.env_cfg import StandingEnvCfg


@configclass
class StandingHIMEnvCfg(StandingEnvCfg):
    observation_space: int = 271        # 6 * 45 + 1 height command
    state_space: int = CRITIC_OBS_DIM + 1


class StandingHIMEnv(StandingEnv):
    cfg: StandingHIMEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        if CRITIC_OBS_DIM != 91:
            raise ValueError(f"Expected 91 critic features, got {CRITIC_OBS_DIM}")
        self.history = HIMHistory(
            num_envs=self.num_envs,
            obs_dim=45,
            history_steps=6,
            device=self.device,
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        if hasattr(self, "history"):
            self.history.reset(env_ids)

    def _get_observations(self):
        current_obs = super()._get_observations()["policy"]
        history_obs = self.history.push(current_obs)

        critic_obs = build_critic_observation(
            robot=self.robot,
            policy_obs=current_obs,
            foot_body_ids=self.foot_body_ids,
            env_origins=self.scene.env_origins,
        )
        assert history_obs.shape == (self.num_envs, 270)
        assert critic_obs.shape == (self.num_envs, 91)

        # Fixed-scale encoding, NOT min/max normalization or clipping.
        # This single command is constant throughout a training run.
        height_norm = (
            self.height_command - float(self.cfg.middle_base_height)
        ) / 0.10
        assert torch.isfinite(height_norm).all()

        policy = torch.cat((history_obs, height_norm), dim=1)
        critic = torch.cat((critic_obs, height_norm), dim=1)
        assert policy.shape == (self.num_envs, 271)
        assert critic.shape == (self.num_envs, 92)
        return {"policy": policy, "critic": critic}
