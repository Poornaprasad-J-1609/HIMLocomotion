"""Newton flat walking using the existing Grallator action and robot classes.

This class deliberately inherits the standing robot setup and 12D action path.
It DOES NOT modify standing sources. All reward terms are per-second rates.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch

from .reward_cfg import load_reward_cfg_from_env
from .rewards import compute_flat_walking_terms
from grallator_rl.tasks.standing_direct.config.grallator.env import StandingEnv
from .env_cfg import WalkingEnvCfg


class WalkingEnv(StandingEnv):
    cfg: WalkingEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        # Set this BEFORE super().__init__ in case environment setup computes rewards.
        self.walking_reward_cfg = load_reward_cfg_from_env()
        super().__init__(cfg, render_mode=render_mode, **kwargs)
        self._command_resample_steps = max(
            1, round(float(self.cfg.command_resample_time_s) / float(self.step_dt))
        )
        self._sample_commands(torch.arange(self.num_envs, device=self.device))

        # Accumulate foot-airtime over training, not just one simulation step.
        names = list(self.contact_sensor.sensor_names)
        self._audit_foot_ids = [
            names.index(name)
            for name in ("FR_foot", "FL_foot", "BR_foot", "BL_foot")
        ]
        self._audit_air = torch.zeros(
            self.num_envs, 4, device=self.device
        )
        self._audit_steps = torch.zeros(
            self.num_envs, device=self.device
        )
        self._audit_liftoffs = torch.zeros(
            self.num_envs, device=self.device
        )
        self._audit_previous_contact = torch.ones(
            self.num_envs, 4, dtype=torch.bool, device=self.device
        )
        print(
            "Grallator walking: action scales unchanged, "
            f"commands resample every {self._command_resample_steps} control steps; "
            f"walking reward config={self.walking_reward_cfg}"
        )

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        # Independent per-environment commands in base frame.
        r = torch.rand(n, 3, device=self.device)
        lows = torch.tensor(
            [self.cfg.cmd_vx_min, self.cfg.cmd_vy_min, self.cfg.cmd_yaw_min],
            device=self.device,
        )
        highs = torch.tensor(
            [self.cfg.cmd_vx_max, self.cfg.cmd_vy_max, self.cfg.cmd_yaw_max],
            device=self.device,
        )
        commands = lows + (highs - lows) * r
        # Include a stationary task to retain standing behavior.
        stand_mask = torch.rand(n, device=self.device) < self.cfg.stand_fraction
        commands[stand_mask] = 0.0
        self.velocity_commands[env_ids] = commands

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # Keep command transitions independent of PPO action computation.
        if hasattr(self, "_command_resample_steps"):
            t = self.episode_length_buf
            resample_ids = torch.nonzero(
                (t > 0) & ((t % self._command_resample_steps) == 0),
                as_tuple=False,
            ).flatten()
            self._sample_commands(resample_ids)
        # Preserves middle_joint_pos + action_scales * clipped actions.
        super()._pre_physics_step(actions)

    def _get_rewards(self) -> torch.Tensor:
        base_height = (
            self.robot.data.root_pos_w.torch[:, 2]
            - self.scene.env_origins[:, 2]
        )
        base_lin_vel = self.robot.data.root_lin_vel_b.torch
        base_ang_vel = self.robot.data.root_ang_vel_b.torch
        undesired = self._contact_mask()
        terms = compute_flat_walking_terms(
            cfg=self.walking_reward_cfg,
            base_lin_vel_b=base_lin_vel,
            base_ang_vel_b=base_ang_vel,
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
                raise RuntimeError(f"Walking reward '{name}' invalid shape or values")
            # Walking reward functions return reward RATES; multiply by step_dt ONCE.
            contribution = rate * self.step_dt
            reward += contribution
            if name not in self.reward_sums:
                self.reward_sums[name] = torch.zeros(self.num_envs, device=self.device)
            self.reward_sums[name] += contribution

        xy_err = torch.linalg.vector_norm(
            base_lin_vel[:, :2] - self.velocity_commands[:, :2], dim=1
        )
        yaw_err = (base_ang_vel[:, 2] - self.velocity_commands[:, 2]).abs()
        log = self.extras.setdefault("log", {})
        log["Diagnostics/xy_vel_error_mps"] = xy_err.mean()
        log["Diagnostics/yaw_error_radps"] = yaw_err.mean()

        # Actual body-frame velocities
        log["Diagnostics/actual_vx_mps"] = base_lin_vel[:, 0].mean()
        log["Diagnostics/actual_vy_mps"] = base_lin_vel[:, 1].mean()
        log["Diagnostics/actual_yaw_radps"] = base_ang_vel[:, 2].mean()

        # Absolute velocity-tracking errors
        log["Diagnostics/vx_mae_mps"] = (
            base_lin_vel[:, 0] - self.velocity_commands[:, 0]
        ).abs().mean()

        log["Diagnostics/vy_mae_mps"] = (
            base_lin_vel[:, 1] - self.velocity_commands[:, 1]
        ).abs().mean()

        # Compare commanded and achieved movement magnitudes
        command_speed = torch.linalg.vector_norm(
            self.velocity_commands[:, :2], dim=1
        )
        actual_speed = torch.linalg.vector_norm(
            base_lin_vel[:, :2], dim=1
        )

        log["Diagnostics/command_speed_mps"] = command_speed.mean()
        log["Diagnostics/actual_speed_mps"] = actual_speed.mean()

        # Measure error only when significant movement is requested
        moving = command_speed > 0.10

        log["Diagnostics/moving_xy_error_mps"] = (
            (xy_err * moving).sum()
            / moving.sum().clamp_min(1)
        )

        log["Diagnostics/moving_command_pct"] = (
            moving.float().mean() * 100.0
        )
        log["Diagnostics/command_vx_mps"] = self.velocity_commands[:, 0].mean()
        log["Diagnostics/command_vy_mps"] = self.velocity_commands[:, 1].mean()
        log["Diagnostics/command_yaw_radps"] = self.velocity_commands[:, 2].mean()
        log["Diagnostics/height_error_cm"] = (
            (base_height - self.height_command[:, 0]).abs().mean() * 100.0
        )
        log["Diagnostics/action_near_limit_pct"] = (
            (self.actions.abs() > 0.95).float().mean() * 100.0
        )
        log["Diagnostics/undesired_contact_pct"] = undesired.float().mean() * 100.0

        # Per-foot contact diagnostics, using a 2 N contact threshold.
        foot_names = ("FR_foot", "FL_foot", "BR_foot", "BL_foot")
        sensor_names = list(self.contact_sensor.sensor_names)
        foot_ids = [sensor_names.index(name) for name in foot_names]

        force = self.contact_sensor.data.net_normal_forces_w.torch[
            :, foot_ids, :
        ]
        foot_force_n = torch.linalg.vector_norm(force, dim=-1)
        feet_contact = foot_force_n > 2.0

        log["Diagnostics/feet_contact_count"] = (
            feet_contact.float().sum(dim=1).mean()
        )
        log["Diagnostics/all_feet_contact_pct"] = (
            feet_contact.all(dim=1).float().mean() * 100.0
        )
        log["Diagnostics/foot_no_contact_pct"] = (
            (~feet_contact).float().mean() * 100.0
        )
        log["Diagnostics/mean_foot_force_n"] = foot_force_n.mean()

        # Ignore the first/last six control steps near episode resets.
        valid = (
            (self.episode_length_buf > 6)
            & (self.episode_length_buf < self.max_episode_length - 6)
            & (torch.linalg.vector_norm(
                self.velocity_commands[:, :2], dim=1
            ) > 0.10)
        )

        forces = self.contact_sensor.data.net_normal_forces_w.torch[
            :, self._audit_foot_ids, :
        ]
        contact = torch.linalg.vector_norm(forces, dim=-1) > 2.0

        self._audit_air += (
            (~contact) & valid[:, None]
        ).float()
        self._audit_steps += valid.float()

        liftoff = (
            self._audit_previous_contact
            & (~contact)
            & valid[:, None]
        )
        self._audit_liftoffs += liftoff.float().sum(dim=1)
        self._audit_previous_contact.copy_(contact)

        denominator = (self._audit_steps.sum() * 4).clamp_min(1.0)

        log["Diagnostics/airborne_pct_cumulative"] = (
            100.0 * self._audit_air.sum() / denominator
        )
        log["Diagnostics/liftoffs_per_1000_foot_steps"] = (
            1000.0 * self._audit_liftoffs.sum() / denominator
        )
        return reward

    def _reset_idx(self, env_ids: Sequence[int] | None):
        # Parent resets MIDDLE_POSE and zeroes commands; then sample walking commands.
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)
        if hasattr(self, "velocity_commands"):
            self._sample_commands(env_ids)
