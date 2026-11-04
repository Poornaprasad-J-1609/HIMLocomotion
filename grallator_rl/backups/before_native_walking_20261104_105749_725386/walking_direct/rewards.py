"""Vectorized, Kit-less flat-ground rewards for a future Grallator walking env.

Every output is a per-second reward rate with shape [num_envs].
The environment must multiply by step_dt exactly once, as its standing env does.
"""
from __future__ import annotations

import torch

from .reward_cfg import WalkingRewardCfg


def compute_flat_walking_terms(
    *,
    cfg: WalkingRewardCfg,
    base_lin_vel_b: torch.Tensor,         # [N,3], BODY frame
    base_ang_vel_b: torch.Tensor,         # [N,3], BODY frame
    projected_gravity_b: torch.Tensor,    # [N,3]
    velocity_commands: torch.Tensor,     # [N,3] = vx, vy, yaw rate, BODY frame
    base_height: torch.Tensor,           # [N], world-relative base z
    target_height: torch.Tensor,         # [N]
    joint_vel: torch.Tensor,             # [N,12]
    actions: torch.Tensor,               # [N,12], normalized
    previous_actions: torch.Tensor,      # [N,12]
    undesired_contact: torch.Tensor,     # [N] bool or float
) -> dict[str, torch.Tensor]:
    n = actions.shape[0]
    for name, tensor, shape in (
        ("base_lin_vel_b", base_lin_vel_b, (n, 3)),
        ("base_ang_vel_b", base_ang_vel_b, (n, 3)),
        ("projected_gravity_b", projected_gravity_b, (n, 3)),
        ("velocity_commands", velocity_commands, (n, 3)),
        ("base_height", base_height, (n,)),
        ("target_height", target_height, (n,)),
        ("joint_vel", joint_vel, actions.shape),
        ("previous_actions", previous_actions, actions.shape),
        ("undesired_contact", undesired_contact, (n,)),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"{name}: expected {shape}, got {tuple(tensor.shape)}")

    xy_error_sq = (
        base_lin_vel_b[:, :2] - velocity_commands[:, :2]
    ).square().sum(dim=-1)
    yaw_error_sq = (base_ang_vel_b[:, 2] - velocity_commands[:, 2]).square()
    height_error_sq = (base_height - target_height).square()

    return {
        "track_xy": cfg.track_xy * torch.exp(-xy_error_sq / cfg.xy_sigma**2),
        "track_yaw": cfg.track_yaw * torch.exp(-yaw_error_sq / cfg.yaw_sigma**2),
        "height": cfg.height * torch.exp(-height_error_sq / cfg.height_sigma**2),
        "upright": -cfg.upright * projected_gravity_b[:, :2].square().sum(dim=-1),
        "vertical_velocity": -cfg.vertical_velocity * base_lin_vel_b[:, 2].square(),
        "joint_velocity": -cfg.joint_velocity * joint_vel.square().mean(dim=-1),
        "action_rate": -cfg.action_rate * (actions - previous_actions).square().mean(dim=-1),
        "action_magnitude": -cfg.action_magnitude * actions.square().mean(dim=-1),
        "undesired_contact": -cfg.undesired_contact * undesired_contact.to(actions.dtype),
    }
