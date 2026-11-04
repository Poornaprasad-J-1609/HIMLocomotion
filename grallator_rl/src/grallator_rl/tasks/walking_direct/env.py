"""Normal Grallator walking DirectRLEnv: Newton + HIM/PPO, no reward-search system.

Inherit the proven standing robot/action/reset implementation; change only
velocity commands and walking reward terms. All weights live in reward_cfg.py.
"""
from __future__ import annotations

from collections.abc import Sequence
import torch

from grallator_rl.tasks.standing_direct.config.grallator.env import StandingEnv
from .env_cfg import WalkingEnvCfg
from .rewards import compute_walking_terms, compute_feet_air_time_rate, compute_optional_terms
from .gait_state import GaitState, FOOT_NAMES, contact_from_forces
from .diagnostics import WalkingDiagnostics


class WalkingEnv(StandingEnv):
    cfg: WalkingEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._command_resample_steps = max(
            1, round(float(self.cfg.command_resample_time_s) / float(self.step_dt))
        )
        sensor_names = list(self.contact_sensor.sensor_names)
        self._foot_sensor_ids = [sensor_names.index(name) for name in FOOT_NAMES]
        self.gait = GaitState(self.num_envs, self.device)
        self.diagnostics = WalkingDiagnostics(self.num_envs, self.device)
        self._previous_joint_velocity = self.robot.data.joint_vel.torch.clone()
        self._previous_vertical_velocity = self.robot.data.root_lin_vel_w.torch[:, 2].clone()
        self._last_resample_step = torch.full_like(self.episode_length_buf, -1)
        self._sample_commands(torch.arange(self.num_envs, device=self.device))
        self.diagnostics.attach_torque(self.robot, self.sim)

    def close(self):
        if hasattr(self, 'diagnostics') and self.diagnostics.torque is not None:
            self.diagnostics.torque.close()
        super().close()

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

    def _get_observations(self):
        if hasattr(self, '_command_resample_steps'):
            steps = self.episode_length_buf
            ids = torch.nonzero(
                (steps > 0) & (steps % self._command_resample_steps == 0)
                & (steps != self._last_resample_step),
                as_tuple=False,
            ).flatten()
            self._sample_commands(ids)
            self._last_resample_step[ids] = steps[ids]
        return super()._get_observations()

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
        rc = self.cfg.rewards
        foot_force = self.contact_sensor.data.net_normal_forces_w.torch[:, self._foot_sensor_ids, :]
        contact = contact_from_forces(foot_force, rc.contact_force_threshold_n)
        transient_free = (self.episode_length_buf > rc.reset_exclusion_steps) & ~self.reset_buf
        moving = self.velocity_commands[:, :2].norm(dim=-1) > rc.moving_threshold_mps
        stable = (self.robot.data.projected_gravity_b.torch[:, 2] < rc.stable_gravity_z_max) & ~undesired
        events = self.gait.update(contact, transient_free & moving & stable, self.step_dt, rc)
        feet_air_time_reward_rate = compute_feet_air_time_rate(
            events['qualified_airtime_s'], rc.feet_air_time, self.step_dt)
        terms['feet_air_time'] = feet_air_time_reward_rate
        joint_vel = self.robot.data.joint_vel.torch
        vertical_vel = self.robot.data.root_lin_vel_w.torch[:, 2]
        joint_acc = (joint_vel - self._previous_joint_velocity) / self.step_dt
        vertical_acc = (vertical_vel - self._previous_vertical_velocity) / self.step_dt
        joint_acc = torch.where(transient_free[:, None], joint_acc, 0.0)
        vertical_acc = torch.where(transient_free, vertical_acc, 0.0)
        self._previous_joint_velocity.copy_(joint_vel)
        self._previous_vertical_velocity.copy_(vertical_vel)
        foot_vel = self.robot.data.body_link_lin_vel_w.torch[:, self.foot_body_ids, :]
        terms.update(compute_optional_terms(
            rc, contact=contact, air_time=self.gait.last_air_time,
            contact_time=self.gait.last_contact_time, foot_velocity_w=foot_vel,
            ang_vel_b=ang_vel, vertical_acceleration=vertical_acc,
            joint_acceleration=joint_acc, joint_error=self.robot.data.joint_pos.torch - self.middle_joint_pos,
            joint_velocity=joint_vel, joint_torque=self.robot.actuators.applied_effort.torch,
            commands=self.velocity_commands))
        # One asynchronous device assertion for all reward/contact signals, avoiding
        # the previous per-term tensor-to-CPU synchronization on every step.
        rates = torch.stack(list(terms.values()))
        if rates.shape != (len(terms), self.num_envs):
            raise RuntimeError('Unexpected reward tensor shape')
        torch._assert_async(torch.isfinite(rates).all() & torch.isfinite(foot_force).all(),
                            'Nonfinite walking reward/contact signal')
        reward = rates.sum(0) * self.step_dt
        for name, rate in terms.items():
            if name not in self.reward_sums:
                self.reward_sums[name] = torch.zeros(self.num_envs, device=self.device)
            self.reward_sums[name] += rate * self.step_dt
        self.extras['log'] = self.diagnostics.snapshot(
            terms=terms, commands=self.velocity_commands, lin_vel=lin_vel, ang_vel=ang_vel,
            eligible=transient_free & moving, contact=contact, events=events,
            foot_velocity=foot_vel, height_error=base_height - self.height_command[:, 0],
            undesired=undesired, actions=self.actions, fallen=self.reset_terminated)
        self.extras['log']['Walking/Tracking/reset_excluded_fraction'] = (~transient_free).float().mean()
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if hasattr(self, 'gait') and len(env_ids):
            log = self.extras.setdefault('log', {})
            if self.reward_sums:
                log['Episode_Reward/total'] = torch.stack(list(self.reward_sums.values())).sum(0)[env_ids].mean()
            self.diagnostics.reset(env_ids, log)
            self.gait.reset(env_ids)
            self._previous_joint_velocity[env_ids] = 0
            self._previous_vertical_velocity[env_ids] = 0
            self._last_resample_step[env_ids] = -1
        super()._reset_idx(env_ids)
        if hasattr(self, 'velocity_commands'):
            self._sample_commands(env_ids)
