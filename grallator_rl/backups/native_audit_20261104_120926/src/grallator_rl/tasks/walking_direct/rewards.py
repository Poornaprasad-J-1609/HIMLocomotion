"""Native Grallator walking rewards, following standing_direct/rewards.py.

Each returned tensor is [N] and represents a per-second reward rate.
No rewards are multiplied by step_dt here; the environment handles that once.
"""
from __future__ import annotations

import torch
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
