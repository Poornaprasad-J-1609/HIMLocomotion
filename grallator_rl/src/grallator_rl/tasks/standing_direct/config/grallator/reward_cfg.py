"""Centralized standing reward coefficients.

All terms are *rates*, multiplied by environment step_dt exactly once in env.py.
Changing a reward coefficient never changes the 45/270D HIM observation layout.
"""

from isaaclab.utils import configclass


@configclass
class StandingRewardCfg:
    # Rewards (positive when standing is successful).
    upright: float = 0.5
    height: float = 0.5
    height_progress: float = 0.5

    # Gaussian/exponential shaping distances, not command min/max constraints.
    height_tolerance_m: float = 0.04
    height_progress_distance_m: float = 0.08

    # Costs (positive coefficients; rewards.py applies a minus sign).
    foot_height_balance: float = 0.2
    foot_balance_distance_m: float = 0.05

    angular_velocity: float = 0.60
    yaw_velocity: float = 2.0
    base_lin_vel: float = 1.0
    vertical_velocity: float = 0.75
    joint_velocity: float = 0.08

    leg_pose_deviation: float = 0.3  # Reduced: don't fight commanded height.
    leg_symmetry: float = 15.0
    front_back_symmetry: float = 15.0
    
    action_rate: float = 0.80
    action_magnitude: float = 0.01
    
    hip_position: float = 5.0
    hip_velocity: float = 0.10
    
    undesired_contact: float = 2.0

    # A contact is counted only when measured force exceeds this threshold.
    contact_force_threshold_n: float = 2.0
