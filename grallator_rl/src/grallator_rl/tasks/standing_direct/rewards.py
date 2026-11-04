"""Torch-only standing reward terms for Grallator.

This module deliberately does not import Isaac Lab so reward math can be unit
-tested independently on CPU. Only the robot's feet may make support contact;
aggregate non-foot contacts also include robot/obstacle self collisions.
"""

from __future__ import annotations

import torch


def mirrored_leg_error(joint_error: torch.Tensor, symmetry_pairs: list[tuple[int, int]]) -> torch.Tensor:
    """Squared mirrored *offset* error, averaged over thigh/calf pairs.

    The Grallator thigh/calf axes use opposite signs left/right, so the
    *sum* of left and right offsets should approach zero. Hip pairs are not
    included because their sign conventions have not been validated.
    """
    return torch.stack(
        [(joint_error[:, r] + joint_error[:, l]).square() for r, l in symmetry_pairs], dim=1
    ).mean(dim=1)
def front_back_leg_error(joint_error, front_back_pairs):
        return torch.stack(
            [
                (
                    joint_error[:, front]
                    - joint_error[:, back]
                ).square()
                for front, back in front_back_pairs
            ],
            dim=1,
        ).mean(dim=1)


def compute_standing_terms(
    *,
    rewards_cfg,
    base_height: torch.Tensor,
    height_command: torch.Tensor,
    projected_gravity: torch.Tensor,
    base_ang_vel: torch.Tensor,
    base_lin_vel: torch.Tensor,
    foot_heights: torch.Tensor,
    joint_error: torch.Tensor,
    joint_vel: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    hip_ids: list[int],
    leg_ids: list[int],
    symmetry_pairs: list[tuple[int, int]],
    undesired_contact: torch.Tensor,
    front_back_pairs: list[tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Return one [num_envs] *unscaled* reward rate per named term.

    There are no hard limits on height commands. The single supplied height
    is the target for both absolute accuracy and dense progress shaping.
    """
    w = rewards_cfg
    height_error = base_height - height_command
    symmetry_error = mirrored_leg_error(joint_error, symmetry_pairs)
    front_back_error = front_back_leg_error(
        joint_error,
        front_back_pairs,
    )

    # Stillness matters most as we approach the goal, without using a hard
    # minimum or maximum height to activate the cost terms.
    near_goal = torch.exp(-height_error.abs() / w.height_progress_distance_m)

    foot_balance_error = (
        foot_heights - foot_heights.mean(dim=1, keepdim=True)
    ) / w.foot_balance_distance_m
    front_back_error = front_back_leg_error(
    joint_error, front_back_pairs
)

    return {
        "upright": w.upright * torch.exp(-8.0 * projected_gravity[:, :2].square().sum(dim=1)),
        "height": w.height * torch.exp(-(height_error / w.height_tolerance_m).square()),
        "height_progress": w.height_progress * torch.exp(
            -height_error.abs() / w.height_progress_distance_m
        ),
        "foot_height_balance": -w.foot_height_balance * foot_balance_error.square().mean(dim=1),
        "angular_velocity": -w.angular_velocity * base_ang_vel[:, :2].square().sum(dim=1),
        "yaw_velocity": -w.yaw_velocity * base_ang_vel[:, 2].square(),
        "base_lin_vel": -w.base_lin_vel * base_lin_vel[:, :2].square().sum(dim=1),
        "vertical_velocity": -w.vertical_velocity * base_lin_vel[:, 2].square() * near_goal,
        "joint_velocity": -w.joint_velocity * joint_vel.square().mean(dim=1) * near_goal,
        # Crucial: don't force thigh/calf to stay at MIDDLE_POSE when a
        # different height requires flexion/extension. Hips are separate.
        "pose_deviation": -w.leg_pose_deviation * joint_error[:, leg_ids].square().mean(dim=1),
        "leg_symmetry": -w.leg_symmetry * symmetry_error,
        "action_rate": -w.action_rate * (actions - previous_actions).square().mean(dim=1) * near_goal,
        "action_magnitude": -w.action_magnitude * actions.square().mean(dim=1) * near_goal,
        "hip_stability": -(
            w.hip_position * joint_error[:, hip_ids].square().sum(dim=1)
            + w.hip_velocity * joint_vel[:, hip_ids].square().sum(dim=1)
        ),
        "undesired_contact": -w.undesired_contact * undesired_contact.float(),
        "front_back_symmetry": -(
            w.front_back_symmetry * front_back_error
        ),
    }
