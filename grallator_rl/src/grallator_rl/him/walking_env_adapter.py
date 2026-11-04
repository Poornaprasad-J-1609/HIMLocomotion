"""Walking HIM environment: 6x45 history + 1 height = 271 actor, 92 critic.

The observation layout is deliberately identical to standing for checkpoint loading.
The 45D proprioceptive observation now carries NONZERO vx/vy/yaw commands.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils import configclass
from grallator_rl.him.history import HIMHistory
from grallator_rl.him.critic import CRITIC_OBS_DIM, build_critic_observation
from grallator_rl.tasks.walking_direct.env import WalkingEnv
from grallator_rl.tasks.walking_direct.env_cfg import WalkingEnvCfg


@configclass
class WalkingHIMEnvCfg(WalkingEnvCfg):
    observation_space: int = 271
    state_space: int = 92


class WalkingHIMEnv(WalkingEnv):
    cfg: WalkingHIMEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        if CRITIC_OBS_DIM != 91:
            raise RuntimeError(f"Expected original 91D privileged observations; got {CRITIC_OBS_DIM}")
        self.history = HIMHistory(
            num_envs=self.num_envs,
            obs_dim=45,
            history_steps=6,
            device=self.device,
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if hasattr(self, "history"):
            self.history.reset(env_ids)

    def _get_observations(self):
        current_obs = super()._get_observations()["policy"]
        history = self.history.push(current_obs)
        critic_obs = build_critic_observation(
            robot=self.robot,
            policy_obs=current_obs,
            foot_body_ids=self.foot_body_ids,
            env_origins=self.scene.env_origins,
        )
        if history.shape != (self.num_envs, 270):
            raise RuntimeError(f"Expected history [N,270], got {history.shape}")
        if critic_obs.shape != (self.num_envs, 91):
            raise RuntimeError(f"Expected critic [N,91], got {critic_obs.shape}")
        height_norm = (
            self.height_command - float(self.cfg.middle_base_height)
        ) / 0.10
        policy = torch.cat((history, height_norm), dim=1)
        critic = torch.cat((critic_obs, height_norm), dim=1)
        assert policy.shape == (self.num_envs, 271)
        assert critic.shape == (self.num_envs, 92)
        return {"policy": policy, "critic": critic}
