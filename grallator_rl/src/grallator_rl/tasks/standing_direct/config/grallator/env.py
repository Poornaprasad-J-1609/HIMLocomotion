"""Grallator Newton standing environment.

Responsibilities:
- exact MIDDLE_POSE resets and one fixed target height
- 12D joint-offset actions, with smaller hip action range
- original deployable 45D proprioception
- contact sensing, reward accumulation and reset diagnostics

Reward mathematics lives in ``grallator_rl.tasks.standing_direct.rewards``.
HIM history and 271D/92D adaptation live in ``grallator_rl.him``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_from_euler_xyz

from grallator.config.observations import ACTION_DIM, OBS_DIM, build_policy_observation
from grallator_rl.tasks.standing_direct.rewards import (compute_standing_terms, mirrored_leg_error,front_back_leg_error)

if TYPE_CHECKING:
    from .env_cfg import StandingEnvCfg


class StandingEnv(DirectRLEnv):
    cfg: StandingEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.robot = self.scene["robot"]
        self.contact_sensor = self.scene["contact_forces"]
        if self.robot.num_joints != ACTION_DIM:
            raise ValueError(f"Expected {ACTION_DIM} joints; found {self.robot.num_joints}")

        # The model's actual tensor joint ordering is authoritative.
        joint_idx = {name: i for i, name in enumerate(self.robot.joint_names)}
        expected = set(joint_idx)
        configured = set(self.cfg.middle_joint_pos)
        if expected != configured:
            raise ValueError(
                f"MIDDLE_POSE joint-name mismatch: missing={expected - configured}, "
                f"extra={configured - expected}"
            )

        # Per-environment MIDDLE_POSE reference, independent of USD defaults.
        self.middle_joint_pos = self.robot.data.default_joint_pos.torch.clone()
        for name, angle in self.cfg.middle_joint_pos.items():
            self.middle_joint_pos[:, joint_idx[name]] = float(angle)

        limits = self.robot.data.joint_pos_limits.torch
        if not ((self.middle_joint_pos >= limits[..., 0]) & (self.middle_joint_pos <= limits[..., 1])).all().item():
            raise ValueError("MIDDLE_POSE violates imported joint position limits")

        self.hip_joint_ids = [joint_idx[name] for name in self.cfg.middle_joint_pos if "_hip_joint" in name]
        self.leg_joint_ids = [i for i in range(ACTION_DIM) if i not in self.hip_joint_ids]
        if len(self.hip_joint_ids) != 4 or len(self.leg_joint_ids) != 8:
            raise ValueError("Grallator needs 4 hip joints and 8 thigh/calf joints")

        self.symmetry_pairs = [
            (joint_idx["FR_thigh_joint"], joint_idx["FL_thigh_joint"]),
            (joint_idx["FR_calf_joint"], joint_idx["FL_calf_joint"]),
            (joint_idx["BR_thigh_joint"], joint_idx["BL_thigh_joint"]),
            (joint_idx["BR_calf_joint"], joint_idx["BL_calf_joint"]),
        ]
        self.front_back_pairs = [
            (joint_idx["FR_thigh_joint"], joint_idx["BR_thigh_joint"]),
            (joint_idx["FR_calf_joint"], joint_idx["BR_calf_joint"]),
            (joint_idx["FL_thigh_joint"], joint_idx["BL_thigh_joint"]),
            (joint_idx["FL_calf_joint"], joint_idx["BL_calf_joint"]),
        ]

        self.action_scales = torch.full(
            (1, ACTION_DIM), self.cfg.leg_action_scale, device=self.device
        )
        self.action_scales[:, self.hip_joint_ids] = self.cfg.hip_action_scale

        foot_names = ["FR_foot", "FL_foot", "BR_foot", "BL_foot"]
        missing_foot_links = set(foot_names) - set(self.robot.body_names)
        if missing_foot_links:
            raise RuntimeError(f"Robot foot links missing: {sorted(missing_foot_links)}")
        self.foot_body_ids = [self.robot.body_names.index(name) for name in foot_names]

        # Sensor indices are independent of the robot body's ordering.
        sensor_names = list(self.contact_sensor.sensor_names)
        missing_sensor_feet = set(foot_names) - set(sensor_names)
        if missing_sensor_feet:
            raise RuntimeError(f"Contact sensor did not match foot links: {sorted(missing_sensor_feet)}")
        self.undesired_contact_ids = [
            i for i, name in enumerate(sensor_names) if name not in set(foot_names)
        ]
        if not self.undesired_contact_ids:
            raise RuntimeError("Contact sensor has no non-foot rigid bodies to monitor")
        print("MIDDLE_POSE joint order:", list(self.robot.joint_names))
        print("Contact sensor body names:", sensor_names)
        print("Non-foot contact bodies:", [sensor_names[i] for i in self.undesired_contact_ids])
        print(f"MIDDLE_POSE reset z={self.cfg.middle_base_height:.3f} m")
        print(f"Single commanded standing z={self.cfg.target_base_height:.3f} m")

        self.actions = torch.zeros(self.num_envs, ACTION_DIM, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)
        self.joint_targets = self.middle_joint_pos.clone()
        self.velocity_commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.height_command = torch.full(
            (self.num_envs, 1), float(self.cfg.target_base_height), device=self.device
        )

        # Automatically constructed from the shared reward implementation.
        self.reward_sums: dict[str, torch.Tensor] = {}

    def _pre_physics_step(self, actions: torch.Tensor):
        if actions.shape != self.actions.shape:
            raise ValueError(f"Expected action shape {tuple(self.actions.shape)}, got {tuple(actions.shape)}")
        self.previous_actions.copy_(self.actions)
        self.actions.copy_(actions.clamp(-1.0, 1.0))
        desired_q = self.middle_joint_pos + self.action_scales * self.actions
        limits = self.robot.data.joint_pos_limits.torch
        self.joint_targets = torch.clamp(desired_q, limits[..., 0], limits[..., 1])

    def _apply_action(self):
        command = self.robot.actuators.target_command
        command.set_position_index(value=self.joint_targets)
        command.set_velocity_index(value=torch.zeros_like(self.joint_targets))
        command.set_effort_index(value=torch.zeros_like(self.joint_targets))

    def _get_observations(self):
        joint_position_error = self.robot.data.joint_pos.torch - self.middle_joint_pos
        obs = build_policy_observation(
            base_ang_vel=self.robot.data.root_ang_vel_b.torch,
            projected_gravity=self.robot.data.projected_gravity_b.torch,
            velocity_commands=self.velocity_commands,
            previous_action=self.actions,
            joint_position_error=joint_position_error,
            joint_vel=self.robot.data.joint_vel.torch,
        )
        if obs.shape != (self.num_envs, OBS_DIM):
            raise RuntimeError(f"Expected policy [N,{OBS_DIM}], got {tuple(obs.shape)}")
        return {"policy": obs}

    def _contact_mask(self) -> torch.Tensor:
        # Newton/Isaac Lab ContactSensor reports per-rigid-body normal force.
        # This is an aggregate contact penalty, not ground-only filtering.
        forces = self.contact_sensor.data.net_normal_forces_w.torch
        expected = (self.num_envs, len(self.contact_sensor.sensor_names), 3)
        if forces.shape != expected:
            raise RuntimeError(f"Expected contact force shape {expected}, got {tuple(forces.shape)}")
        nonfoot_forces = forces[:, self.undesired_contact_ids, :]
        force_n = torch.linalg.vector_norm(nonfoot_forces, dim=-1)
        return (force_n > self.cfg.rewards.contact_force_threshold_n).any(dim=1)

    def _get_rewards(self):
        joint_pos = self.robot.data.joint_pos.torch
        joint_vel = self.robot.data.joint_vel.torch
        joint_error = joint_pos - self.middle_joint_pos

        base_height = self.robot.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        target_height = self.height_command[:, 0]
        projected_gravity = self.robot.data.projected_gravity_b.torch
        base_ang_vel = self.robot.data.root_ang_vel_b.torch
        base_lin_vel = self.robot.data.root_lin_vel_b.torch
        foot_heights = self.robot.data.body_link_pos_w.torch[:, self.foot_body_ids, 2]
        undesired_contact = self._contact_mask()
        

        terms = compute_standing_terms(
            rewards_cfg=self.cfg.rewards,
            base_height=base_height,
            height_command=target_height,
            projected_gravity=projected_gravity,
            base_ang_vel=base_ang_vel,
            base_lin_vel=base_lin_vel,
            foot_heights=foot_heights,
            joint_error=joint_error,
            joint_vel=joint_vel,
            actions=self.actions,
            previous_actions=self.previous_actions,
            hip_ids=self.hip_joint_ids,
            leg_ids=self.leg_joint_ids,
            symmetry_pairs=self.symmetry_pairs,
            undesired_contact=undesired_contact,
            front_back_pairs=self.front_back_pairs,
        )

        reward = torch.zeros(self.num_envs, device=self.device)
        for name, rate in terms.items():
            if rate.shape != (self.num_envs,):
                raise ValueError(f"Reward '{name}' has unexpected shape {tuple(rate.shape)}")
            contribution = rate * self.step_dt  # ONCE for every term
            reward += contribution
            if name not in self.reward_sums:
                self.reward_sums[name] = torch.zeros(self.num_envs, device=self.device)
            self.reward_sums[name] += contribution

        height_error = base_height - target_height
        sym = mirrored_leg_error(joint_error, self.symmetry_pairs)
        log = self.extras.setdefault("log", {})
        log["Diagnostics/height_target_m"] = target_height.mean()
        log["Diagnostics/base_height_m"] = base_height.mean()
        log["Diagnostics/height_error_cm"] = height_error.abs().mean() * 100.0
        log["Diagnostics/symmetry_rmse_deg"] = torch.sqrt(sym.mean()) * (180.0 / torch.pi)
        log["Diagnostics/action_rms"] = torch.sqrt(self.actions.square().mean())
        log["Diagnostics/action_near_limit_pct"] = (self.actions.abs() > 0.95).float().mean() * 100.0
        log["Diagnostics/undesired_contact_pct"] = undesired_contact.float().mean() * 100.0
        log["Diagnostics/hip_error_deg"] = (
            joint_error[:, self.hip_joint_ids].abs().mean() * (180.0 / torch.pi)
        )
        front_back_sym = front_back_leg_error(
            joint_error, self.front_back_pairs
        )

        log["Diagnostics/front_back_symmetry_rmse_deg"] = (
            torch.sqrt(front_back_sym.mean()) * (180.0 / torch.pi)
        )
        return reward

    def _get_dones(self):
        gravity = self.robot.data.projected_gravity_b.torch
        base_height = self.robot.data.root_pos_w.torch[:, 2] - self.scene.env_origins[:, 2]
        fallen = (base_height < self.cfg.fall_termination_height) | (gravity[:, 2] > -0.65)
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        return fallen, timeout

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)

        # Log episode totals BEFORE clearing counters or resetting physics.
        if hasattr(self, "reward_sums") and self.reward_sums and len(env_ids) > 0:
            log = self.extras.setdefault("log", {})
            for name, values in self.reward_sums.items():
                log[f"Episode_Reward/{name}"] = values[env_ids].mean()
                values[env_ids] = 0.0
            log["Metrics/fall_fraction"] = self.reset_terminated[env_ids].float().mean()

        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0
        self.joint_targets[env_ids] = self.middle_joint_pos[env_ids]
        self.velocity_commands[env_ids] = 0.0
        self.height_command[env_ids, 0] = float(self.cfg.target_base_height)

        # Exact MIDDLE_POSE: no random joint offsets or roll/pitch disturbance.
        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_pose[:, 2] = self.scene.env_origins[env_ids, 2] + self.cfg.middle_base_height
        n = len(env_ids)
        zeros = torch.zeros(n, device=self.device)
        root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, zeros)
        root_velocity = torch.zeros_like(self.robot.data.default_root_vel.torch[env_ids])
        joint_pos = self.middle_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(self.robot.data.default_joint_vel.torch[env_ids])

        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
