"""Native Grallator walking rewards, following standing_direct/rewards.py.

Each returned tensor is [N] and represents a per-second reward rate.
No rewards are multiplied by step_dt here; the environment handles that once.
"""
from __future__ import annotations

import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .reward_cfg import WalkingRewardCfg


def compute_walking_terms(
    *,
    rewards_cfg: WalkingRewardCfg,
    base_lin_vel_b: torch.Tensor,
    base_ang_vel_b: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    velocity_commands: torch.Tensor,
    base_height: torch.Tensor,
    target_height: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    undesired_contact: torch.Tensor,
) -> dict[str, torch.Tensor]:
    n = actions.shape[0]
    if (base_lin_vel_b.shape != (n, 3)
            or base_ang_vel_b.shape != (n, 3)
            or projected_gravity_b.shape != (n, 3)
            or velocity_commands.shape != (n, 3)
            or base_height.shape != (n,)
            or target_height.shape != (n,)
            or joint_vel.shape != actions.shape
            or previous_actions.shape != actions.shape
            or undesired_contact.shape != (n,)):
        raise ValueError('Unexpected walking-reward input tensor shapes')

    xy_error_sq = (
        base_lin_vel_b[:, :2] - velocity_commands[:, :2]
    ).square().sum(dim=-1)
    yaw_error_sq = (
        base_ang_vel_b[:, 2] - velocity_commands[:, 2]
    ).square()
    height_error_sq = (base_height - target_height).square()

    return {
        'track_xy': rewards_cfg.track_xy * torch.exp(-xy_error_sq / rewards_cfg.xy_sigma**2),
        'track_yaw': rewards_cfg.track_yaw * torch.exp(-yaw_error_sq / rewards_cfg.yaw_sigma**2),
        'height': rewards_cfg.height * torch.exp(-height_error_sq / rewards_cfg.height_sigma**2),
        'upright': -rewards_cfg.upright * projected_gravity_b[:, :2].square().sum(dim=-1),
        'vertical_velocity': -rewards_cfg.vertical_velocity * base_lin_vel_b[:, 2].square(),
        'joint_velocity': -rewards_cfg.joint_velocity * joint_vel.square().mean(dim=-1),
        'action_rate': -rewards_cfg.action_rate * (actions - previous_actions).square().mean(dim=-1),
        'action_magnitude': -rewards_cfg.action_magnitude * actions.square().mean(dim=-1),
        'undesired_contact': -rewards_cfg.undesired_contact * undesired_contact.to(actions.dtype),
    }


def compute_feet_air_time_rate(qualified_airtime_s, weight, step_dt):
    """[N,4] completed valid swings -> [N] rate; integral = weight * sum(s)."""
    if step_dt <= 0 or qualified_airtime_s.ndim != 2 or qualified_airtime_s.shape[1] != 4:
        raise ValueError('Expected positive dt and [N,4] qualified swing times')
    return weight * qualified_airtime_s.sum(-1) / step_dt


def diagonal_trot_cost(contact):
    """FR, FL, BR, BL: synchronize FL+BR / FR+BL; oppose the two pairs."""
    return ((contact[:, 1] != contact[:, 2]).float()
            + (contact[:, 0] != contact[:, 3]).float()
            + (contact[:, 0] == contact[:, 1]).float()) / 3.0


def compute_optional_terms(cfg, *, contact, air_time, contact_time, foot_velocity_w,
                           ang_vel_b, vertical_acceleration, joint_acceleration,
                           joint_error, joint_velocity, joint_torque, commands):
    """Pure per-second costs; torque math requires trustworthy joint-side efforts.

    Acceleration uses control-interval finite differences with reset masking in
    env.py. Power is sum(abs(tau*qdot)) in joint coordinates, not battery power.
    env_cfg rejects nonzero torque/power weights with current shadow-PD telemetry.
    No clearance/impact proxy: sole geometry and peak impulse are not validated.
    """
    moving = commands[:, :2].norm(dim=-1) > cfg.moving_threshold_mps
    return {
        'stance_slip': -cfg.stance_slip * (foot_velocity_w[..., :2].square().sum(-1) * contact).sum(-1),
        'trot_sync': -cfg.trot_sync * diagonal_trot_cost(contact) * moving,
        'time_imbalance': -cfg.time_imbalance * (air_time.var(-1, unbiased=False)
                                               + contact_time.var(-1, unbiased=False)) * moving,
        'roll_pitch_velocity': -cfg.roll_pitch_velocity * ang_vel_b[:, :2].square().sum(-1),
        'vertical_acceleration': -cfg.vertical_acceleration * vertical_acceleration.square(),
        'joint_acceleration': -cfg.joint_acceleration * joint_acceleration.square().sum(-1),
        'joint_torque': -cfg.joint_torque * joint_torque.square().sum(-1),
        'joint_position': -cfg.joint_position * joint_error.square().sum(-1)
                          * torch.where(moving, 1.0, cfg.standing_pose_multiplier),
        'mechanical_power': -cfg.mechanical_power * (joint_torque * joint_velocity).abs().sum(-1),
    }
