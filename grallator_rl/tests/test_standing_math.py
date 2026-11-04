"""CPU-only checks for the new reward math (no simulator/GPU required)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


source = Path(__file__).resolve().parents[1] / "src/grallator_rl/tasks/standing_direct/rewards.py"
spec = importlib.util.spec_from_file_location("standing_math_under_test", source)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


W = SimpleNamespace(
    upright=0.7, height=2.0, height_progress=0.5,
    height_tolerance_m=0.04, height_progress_distance_m=0.12,
    foot_height_balance=0.5, foot_balance_distance_m=0.03,
    angular_velocity=0.4, yaw_velocity=0.25, base_lin_vel=0.5,
    vertical_velocity=0.5, joint_velocity=0.04,
    leg_pose_deviation=0.15, leg_symmetry=5.0, action_rate=0.3,
    action_magnitude=0.04, hip_position=3.7, hip_velocity=0.05,
    undesired_contact=2.0,
)


def inputs(target=0.50, base_height=0.50, contact=False):
    n = 3
    return dict(
        rewards_cfg=W,
        base_height=torch.full((n,), base_height),
        height_command=torch.full((n,), target),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]] * n),
        base_ang_vel=torch.zeros(n, 3),
        base_lin_vel=torch.zeros(n, 3),
        foot_heights=torch.zeros(n, 4),
        joint_error=torch.zeros(n, 12),
        joint_vel=torch.zeros(n, 12),
        actions=torch.zeros(n, 12),
        previous_actions=torch.zeros(n, 12),
        hip_ids=[0, 3, 6, 9],
        leg_ids=[i for i in range(12) if i not in [0, 3, 6, 9]],
        symmetry_pairs=[(1, 4), (2, 5), (7, 10), (8, 11)],
        undesired_contact=torch.full((n,), contact),
    )


def test_height_reward_peaks_at_the_requested_height():
    at = module.compute_standing_terms(**inputs(target=0.50, base_height=0.50))
    far = module.compute_standing_terms(**inputs(target=0.50, base_height=0.42))
    assert torch.all(at["height"] > far["height"])
    assert torch.all(at["height_progress"] > far["height_progress"])
    assert at["height"].shape == (3,)


def test_undesired_contact_has_negative_rate_only_on_contact():
    clear = module.compute_standing_terms(**inputs(contact=False))
    bad = module.compute_standing_terms(**inputs(contact=True))
    assert torch.all(clear["undesired_contact"] == 0)
    assert torch.all(bad["undesired_contact"] == -2.0)


def test_nonzero_leg_offset_is_penalized_weakly_but_not_forced_to_zero():
    values = inputs()
    values["joint_error"][:, 1] = 0.2
    terms = module.compute_standing_terms(**values)
    assert (terms["pose_deviation"] < 0).all()
    assert torch.isfinite(torch.stack(list(terms.values()))).all()


def test_mirrored_leg_offsets_cancel():
    errors = torch.zeros(1, 12)
    errors[:, 1] = -0.1
    errors[:, 4] = +0.1
    sym = module.mirrored_leg_error(errors, [(1, 4)])
    assert sym.item() == 0.0
