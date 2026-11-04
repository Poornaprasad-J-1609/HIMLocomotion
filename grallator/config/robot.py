import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from grallator.config.actuators import GRALLATOR_ACTUATORS

GRALLATOR_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/poornaprasad/RL/Quadruped/USD/Grallator_Final/Grallator_Final.usda"
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.428),
        joint_pos={
            "FR_hip_joint": 0.0,
            "FR_thigh_joint": -0.325,
            "FR_calf_joint": -0.675,

            "FL_hip_joint": 0.0,
            "FL_thigh_joint": 0.325,
            "FL_calf_joint": 0.675,

            "BR_hip_joint": 0.0,
            "BR_thigh_joint": -0.325,
            "BR_calf_joint": -0.675,

            "BL_hip_joint": 0.0,
            "BL_thigh_joint": 0.325,
            "BL_calf_joint": 0.675,
        },
        joint_vel={".*": 0.0},
    ),

    actuators=GRALLATOR_ACTUATORS,
)
