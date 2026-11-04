"""Independent walking task. Robot, physics and action scales stay as in StandingEnvCfg."""
from __future__ import annotations

import math
from isaaclab.utils import configclass
from grallator_rl.tasks.standing_direct.config.grallator.env_cfg import StandingEnvCfg
from .reward_cfg import WalkingRewardCfg


@configclass
class WalkingEnvCfg(StandingEnvCfg):
    # Edit commands here. All are in the robot's body frame.
    cmd_vx_min: float = 0.15       # m/s, forward walking curriculum
    cmd_vx_max: float = 0.25
    cmd_vy_min: float = 0.0
    cmd_vy_max: float = 0.0
    cmd_yaw_min: float = 0.0       # rad/s
    cmd_yaw_max: float = 0.0
    stand_fraction: float = 0.10
    command_resample_time_s: float = 3.0

    # Edit reward weights in reward_cfg.py, calculations in rewards.py.
    rewards: WalkingRewardCfg = WalkingRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.validate_weights()
        if self.rewards.joint_torque or self.rewards.mechanical_power:
            raise ValueError('Torque/power costs need validated solver effort; current implicit-actuator '
                             'applied_effort is shadow-PD telemetry. Leave these weights at 0.0.')
        for lo, hi in (
            (self.cmd_vx_min, self.cmd_vx_max),
            (self.cmd_vy_min, self.cmd_vy_max),
            (self.cmd_yaw_min, self.cmd_yaw_max),
        ):
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError('Invalid walking velocity-command interval')
        if not 0 <= self.stand_fraction <= 1:
            raise ValueError('stand_fraction must be within [0, 1]')
        if not math.isfinite(self.command_resample_time_s) or self.command_resample_time_s <= 0:
            raise ValueError('command_resample_time_s must be positive')
