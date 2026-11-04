from isaaclab.actuators import ImplicitActuatorCfg

GRALLATOR_KP = 250.0
GRALLATOR_KD = 4.0
GRALLATOR_MAX_TORQUE = 100.0

GRALLATOR_ACTUATORS = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=[
            ".*_(hip|thigh|calf)_joint"
        ],
        stiffness=GRALLATOR_KP,
        damping=GRALLATOR_KD,
        joint_effort_limit=GRALLATOR_MAX_TORQUE,
    )
}
