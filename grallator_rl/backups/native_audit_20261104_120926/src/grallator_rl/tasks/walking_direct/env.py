"""Normal Grallator walking DirectRLEnv: Newton + HIM/PPO, no reward-search system.

Inherit the proven standing robot/action/reset implementation; change only
velocity commands and walking reward terms. All weights live in reward_cfg.py.
"""
from __future__ import annotations

from collections.abc import Sequence
import torch

from grallator_rl.tasks.standing_direct.config.grallator.env import StandingEnv
from .env_cfg import WalkingEnvCfg
from .rewards import compute_walking_terms


class WalkingEnv(StandingEnv):
    cfg: WalkingEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._command_resample_steps = max(
            1, round(float(self.cfg.command_resample_time_s) / float(self.step_dt))
        )
        foot_names = ('FR_foot', 'FL_foot', 'BR_foot', 'BL_foot')
        sensor_names = list(self.contact_sensor.sensor_names)
        self._foot_sensor_ids = [sensor_names.index(name) for name in foot_names]
        self._air_steps = torch.zeros(self.num_envs, device=self.device)
        self._valid_foot_steps = torch.zeros(self.num_envs, device=self.device)
        self._liftoffs = torch.zeros(self.num_envs, device=self.device)
        self._previous_contact = torch.ones(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        self._sample_commands(torch.arange(self.num_envs, device=self.device))

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if not n:
            return
        lower = torch.tensor(
            [self.cfg.cmd_vx_min, self.cfg.cmd_vy_min, self.cfg.cmd_yaw_min],
            device=self.device,
        )
        upper = torch.tensor(
            [self.cfg.cmd_vx_max, self.cfg.cmd_vy_max, self.cfg.cmd_yaw_max],
            device=self.device,
        )
        commands = lower + (upper - lower) * torch.rand(n, 3, device=self.device)
        standing = torch.rand(n, device=self.device) < self.cfg.stand_fraction
        commands[standing] = 0.0
        self.velocity_commands[env_ids] = commands

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if hasattr(self, '_command_resample_steps'):
            steps = self.episode_length_buf
            ids = torch.nonzero(
                (steps > 0) & (steps % self._command_resample_steps == 0),
                as_tuple=False,
            ).flatten()
            self._sample_commands(ids)
        super()._pre_physics_step(actions)

    def _get_rewards(self) -> torch.Tensor:
        base_height = (
            self.robot.data.root_pos_w.torch[:, 2]
            - self.scene.env_origins[:, 2]
        )
        lin_vel = self.robot.data.root_lin_vel_b.torch
        ang_vel = self.robot.data.root_ang_vel_b.torch
        undesired = self._contact_mask()
        terms = compute_walking_terms(
            rewards_cfg=self.cfg.rewards,
            base_lin_vel_b=lin_vel,
            base_ang_vel_b=ang_vel,
            projected_gravity_b=self.robot.data.projected_gravity_b.torch,
            velocity_commands=self.velocity_commands,
            base_height=base_height,
            target_height=self.height_command[:, 0],
            joint_vel=self.robot.data.joint_vel.torch,
            actions=self.actions,
            previous_actions=self.previous_actions,
            undesired_contact=undesired,
        )
        reward = torch.zeros(self.num_envs, device=self.device)
        for name, rate in terms.items():
            if rate.shape != (self.num_envs,) or not torch.isfinite(rate).all():
                raise RuntimeError(f'Invalid walking reward term: {name}')
            contribution = rate * self.step_dt  # ONCE, as in standing env.py
            reward += contribution
            if name not in self.reward_sums:
                self.reward_sums[name] = torch.zeros(self.num_envs, device=self.device)
            self.reward_sums[name] += contribution

        error_xy = torch.linalg.vector_norm(
            lin_vel[:, :2] - self.velocity_commands[:, :2], dim=1
        )
        error_yaw = (ang_vel[:, 2] - self.velocity_commands[:, 2]).abs()
        target_speed = torch.linalg.vector_norm(self.velocity_commands[:, :2], dim=1)
        actual_speed = torch.linalg.vector_norm(lin_vel[:, :2], dim=1)
        foot_force = self.contact_sensor.data.net_normal_forces_w.torch[
            :, self._foot_sensor_ids, :
        ]
        contact = torch.linalg.vector_norm(foot_force, dim=-1) > self.cfg.rewards.contact_force_threshold_n
        valid = (
            (self.episode_length_buf > 6)
            & (self.episode_length_buf < self.max_episode_length - 6)
            & (target_speed > 0.10)
        )
        self._air_steps += ((~contact) & valid[:, None]).float().sum(dim=1)
        self._valid_foot_steps += 4.0 * valid.float()
        liftoffs = self._previous_contact & (~contact) & valid[:, None]
        self._liftoffs += liftoffs.float().sum(dim=1)
        self._previous_contact.copy_(contact)
        total_foot_steps = self._valid_foot_steps.sum().clamp_min(1.0)

        log = self.extras.setdefault('log', {})
        log['Diagnostics/command_vx_mps'] = self.velocity_commands[:, 0].mean()
        log['Diagnostics/command_vy_mps'] = self.velocity_commands[:, 1].mean()
        log['Diagnostics/command_yaw_radps'] = self.velocity_commands[:, 2].mean()
        log['Diagnostics/actual_vx_mps'] = lin_vel[:, 0].mean()
        log['Diagnostics/actual_vy_mps'] = lin_vel[:, 1].mean()
        log['Diagnostics/actual_yaw_radps'] = ang_vel[:, 2].mean()
        log['Diagnostics/vx_mae_mps'] = (lin_vel[:, 0] - self.velocity_commands[:, 0]).abs().mean()
        log['Diagnostics/vy_mae_mps'] = (lin_vel[:, 1] - self.velocity_commands[:, 1]).abs().mean()
        log['Diagnostics/xy_vel_error_mps'] = error_xy.mean()
        log['Diagnostics/yaw_error_radps'] = error_yaw.mean()
        log['Diagnostics/command_speed_mps'] = target_speed.mean()
        log['Diagnostics/actual_speed_mps'] = actual_speed.mean()
        moving = (target_speed > 0.10).float()
        log['Diagnostics/moving_xy_error_mps'] = (
            (error_xy * moving).sum() / moving.sum().clamp_min(1.0)
        )
        log['Diagnostics/moving_command_pct'] = moving.mean() * 100.0
        log['Diagnostics/height_error_cm'] = (
            base_height - self.height_command[:, 0]
        ).abs().mean() * 100.0
        log['Diagnostics/action_near_limit_pct'] = (
            (self.actions.abs() > 0.95).float().mean() * 100.0
        )
        log['Diagnostics/undesired_contact_pct'] = undesired.float().mean() * 100.0
        log['Diagnostics/feet_contact_count'] = contact.float().sum(dim=1).mean()
        log['Diagnostics/all_feet_contact_pct'] = contact.all(dim=1).float().mean() * 100.0
        log['Diagnostics/airborne_pct_cumulative'] = 100.0 * self._air_steps.sum() / total_foot_steps
        log['Diagnostics/liftoffs_per_1000_foot_steps'] = 1000.0 * self._liftoffs.sum() / total_foot_steps
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if hasattr(self, 'velocity_commands'):
            self._sample_commands(env_ids)
        if hasattr(self, '_previous_contact'):
            self._previous_contact[env_ids] = True
