"""Register only the walking task, without changing existing standing registration."""
import gymnasium as gym


gym.register(
    id="GrallatorRl-HIM-Walking-Direct",
    entry_point="grallator_rl.him.walking_env_adapter:WalkingHIMEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "grallator_rl.him.walking_env_adapter:WalkingHIMEnvCfg",
        "rsl_rl_cfg_entry_point": "grallator_rl.him.him_ppo_cfg:HIMPPORunnerCfg",
        "default_agent": "rsl_rl",
    },
)
