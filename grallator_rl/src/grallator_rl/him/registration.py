import gymnasium as gym


gym.register(
    id="GrallatorRl-HIM-Standing-Direct",

    entry_point=(
        "grallator_rl.him.env_adapter:StandingHIMEnv"
    ),

    disable_env_checker=True,

    kwargs={
        "env_cfg_entry_point": (
            "grallator_rl.him.env_adapter:"
            "StandingHIMEnvCfg"
        ),

        "rsl_rl_cfg_entry_point": (
            "grallator_rl.him.him_ppo_cfg:"
            "HIMPPORunnerCfg"
        ),

        "default_agent": "rsl_rl",
    },
)