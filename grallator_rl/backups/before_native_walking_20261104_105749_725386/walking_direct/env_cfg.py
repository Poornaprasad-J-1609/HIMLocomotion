"""Flat walking configuration that inherits EXACT standing robot, action and sim settings."""
from __future__ import annotations

import math
import os

from isaaclab.utils import configclass

from grallator_rl.tasks.standing_direct.config.grallator.env_cfg import StandingEnvCfg


@configclass
class WalkingEnvCfg(StandingEnvCfg):
    # BODY-FRAME commands (m/s, m/s, rad/s).
    cmd_vx_min: float = float(os.environ.get("GRALLATOR_WALK_VX_MIN", "-0.35"))
    cmd_vx_max: float = float(os.environ.get("GRALLATOR_WALK_VX_MAX", "0.35"))
    cmd_vy_min: float = float(os.environ.get("GRALLATOR_WALK_VY_MIN", "-0.20"))
    cmd_vy_max: float = float(os.environ.get("GRALLATOR_WALK_VY_MAX", "0.20"))
    cmd_yaw_min: float = float(os.environ.get("GRALLATOR_WALK_YAW_MIN", "-0.45"))
    cmd_yaw_max: float = float(os.environ.get("GRALLATOR_WALK_YAW_MAX", "0.45"))
    stand_fraction: float = float(os.environ.get("GRALLATOR_WALK_STAND_FRACTION", "0.25"))  # 25% zero-velocity commands
    command_resample_time_s: float = 3.0

    def __post_init__(self):
        super().__post_init__()
        for lo, hi in (
            (self.cmd_vx_min, self.cmd_vx_max),
            (self.cmd_vy_min, self.cmd_vy_max),
            (self.cmd_yaw_min, self.cmd_yaw_max),
        ):
            if not (math.isfinite(lo) and math.isfinite(hi) and lo <= hi):
                raise ValueError("Invalid walking velocity command interval")
        if not (0 <= self.stand_fraction <= 1):
            raise ValueError("stand_fraction must be between zero and one")
        if not math.isfinite(self.command_resample_time_s) or self.command_resample_time_s <= 0:
            raise ValueError("command_resample_time_s must be positive")
