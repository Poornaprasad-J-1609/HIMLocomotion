
from isaaclab.utils import configclass
from grallator_rl.tasks.standing_direct.config.grallator.reward_cfg import StandingRewardCfg


@configclass
class WalkingRewardCfg(StandingRewardCfg):
    # Inherits contact_force_threshold_n and the two standing height tolerance
    # settings used by StandingEnvCfg.__post_init__. Other inherited standing
    # coefficients are intentionally NOT used by walking/rewards.py.
    track_xy: float = 4.0
    track_yaw: float = 1.0
    xy_sigma: float = 0.15         # m/s, Gaussian velocity tracking width
    yaw_sigma: float = 0.40        # rad/s
    height_sigma: float = 0.04     # m

    upright: float = 0.60
    height: float = 0.50
    vertical_velocity: float = 0.40
    joint_velocity: float = 0.01
    action_rate: float = 0.05
    action_magnitude: float = 0.01
    undesired_contact: float = 2.0
