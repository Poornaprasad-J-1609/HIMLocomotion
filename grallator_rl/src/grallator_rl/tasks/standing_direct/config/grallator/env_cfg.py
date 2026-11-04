"""Grallator single-height standing task on Isaac Lab 3.0 / Newton.

ONE requested height (``target_base_height``); no min/max sampling.
The exact 12-joint MIDDLE_POSE is always the reset/reference configuration.
"""

from __future__ import annotations

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass, replace
from isaaclab_newton.physics import NewtonCfg

from grallator.config.robot import GRALLATOR_CFG
from .reward_cfg import StandingRewardCfg


@configclass
class StandingSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = replace(
        GRALLATOR_CFG,
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # Detect actual contacts from all rigid-body links, not center-height tests.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=1,
        debug_vis=False,
    )


@configclass
class StandingEnvCfg(DirectRLEnvCfg):
    # Base z when reset in exact MIDDLE_POSE (verify against robot collision geometry).
    middle_base_height: float = 0.27

    middle_joint_pos: dict[str, float] = {
        "FR_hip_joint": 0.0,
        "FR_thigh_joint": -0.325,
        "FR_calf_joint": -0.675,
        "FL_hip_joint": 0.0,
        "FL_thigh_joint": +0.325,
        "FL_calf_joint": +0.675,
        "BR_hip_joint": 0.0,
        "BR_thigh_joint": -0.325,
        "BR_calf_joint": -0.675,
        "BL_hip_joint": 0.0,
        "BL_thigh_joint": +0.325,
        "BL_calf_joint": +0.675,
    }

    # Exactly ONE command. Change this value or set GRALLATOR_TARGET_HEIGHT.
    # The uploaded project specified 0.50 m. This is NOT confirmed reachable.
    target_base_height: float = float(os.environ.get("GRALLATOR_TARGET_HEIGHT", "0.41"))

    # Joint-target offsets from MIDDLE_POSE (radians); joint limits still enforced.
    hip_action_scale: float = 0.25
    leg_action_scale: float = 0.25

    # Only used to terminate an already fallen robot, not a commanded height min.
    fall_termination_height: float = 0.09

    rewards: StandingRewardCfg = StandingRewardCfg()

    decimation = 4
    episode_length_s = 12.0
    observation_space = 45
    action_space = 12
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=4,
        physics=NewtonCfg(),
    )

    scene: StandingSceneCfg = StandingSceneCfg(
        num_envs=16,
        env_spacing=2.0,
        replicate_physics=True,
    )

    def __post_init__(self):
        # Required by the USD contact reporter configuration; Newton must still
        # have collision geometry on the links for forces to be detected.
        self.scene.robot.spawn.activate_contact_sensors = True
        if not math.isfinite(self.target_base_height) or self.target_base_height <= 0.0:
            raise ValueError("target_base_height must be one positive, finite value in metres")
        if not math.isfinite(self.middle_base_height) or self.middle_base_height <= 0.0:
            raise ValueError("middle_base_height must be positive and finite")
        if self.hip_action_scale <= 0 or self.leg_action_scale <= 0:
            raise ValueError("Action scales must be positive")
        if self.rewards.height_tolerance_m <= 0 or self.rewards.height_progress_distance_m <= 0:
            raise ValueError("Height reward distances must be positive")
