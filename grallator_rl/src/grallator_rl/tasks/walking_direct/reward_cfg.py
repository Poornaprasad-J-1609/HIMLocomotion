
from isaaclab.utils import configclass
import math
from grallator_rl.tasks.standing_direct.config.grallator.reward_cfg import StandingRewardCfg


@configclass
class WalkingRewardCfg(StandingRewardCfg):
    # Inherits contact_force_threshold_n and the two standing height tolerance
    # settings used by StandingEnvCfg.__post_init__. Other inherited standing
    # coefficients are intentionally NOT used by walking/rewards.py.
    track_xy: float = 7.0
    track_yaw: float = 1.0
    xy_sigma: float = 0.1        # m/s, Gaussian velocity tracking width
    yaw_sigma: float = 0.01        # rad/s
    height_sigma: float = 0.04     # m

    upright: float = 0.60
    height: float = 0.50
    vertical_velocity: float = 0.40
    joint_velocity: float = 0.01
    action_rate: float = 0.05
    action_magnitude: float = 0.01
    undesired_contact: float = 2.0

    # A touchdown earns weight * completed swing seconds, once. env.py converts
    # this event amount to a rate before the shared step_dt integration.
    feet_air_time: float = 2.0
    min_swing_s: float = 0.09
    max_swing_s: float = 0.5
    min_support_feet: int = 2
    moving_threshold_mps: float = 0.10
    reset_exclusion_steps: int = 6
    stable_gravity_z_max: float = -0.85

    # Optional research-task biases: positive cost weights, all off initially.
    stance_slip: float = 0.0
    trot_sync: float = 0.0
    time_imbalance: float = 0.0
    roll_pitch_velocity: float = 0.0
    vertical_acceleration: float = 0.0
    # Runtime-disabled placeholders: implicit-actuator telemetry is shadow PD,
    # not validated solver torque. env_cfg rejects nonzero weights until a
    # trustworthy effort signal is connected. Pure math remains CPU-testable.
    joint_torque: float = 0.0
    joint_acceleration: float = 0.0
    joint_position: float = 0.0
    mechanical_power: float = 0.0
    standing_pose_multiplier: float = 5.0

    def validate_weights(self):
        for name in (
            'track_xy', 'track_yaw', 'height', 'upright', 'vertical_velocity',
            'joint_velocity', 'action_rate', 'action_magnitude', 'undesired_contact',
            'feet_air_time', 'stance_slip', 'trot_sync', 'time_imbalance',
            'roll_pitch_velocity', 'vertical_acceleration', 'joint_torque',
            'joint_acceleration', 'joint_position', 'mechanical_power',
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('xy_sigma', 'yaw_sigma', 'height_sigma', 'contact_force_threshold_n',
                     'moving_threshold_mps', 'standing_pose_multiplier'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not (0 < self.min_swing_s <= self.max_swing_s < float('inf')):
            raise ValueError('Invalid swing duration bounds')
        if self.min_support_feet not in (2, 3, 4) or self.reset_exclusion_steps < 1:
            raise ValueError('Invalid support/reset gate')
        if not -1 <= self.stable_gravity_z_max < 0:
            raise ValueError('Invalid gravity stability gate')
