"""RSL-RL HIM PPO runner configuration for middle-to-height Stage 1."""
from pathlib import Path

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlPpoAlgorithmCfg,
    RslRlOnPolicyRunnerCfg,
)


@configclass
class HIMActorCfg(RslRlMLPModelCfg):
    class_name: str = "grallator_rl.him.rsl_actor:HIMRslActorModel"
    # Resolve the pretrained estimator relative to this project checkout.
    estimator_checkpoint_path: str = str(
        Path(__file__).resolve().parents[3] / "checkpoints" / "him_hio_smoke.pt"
    )


@configclass
class HIMAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "grallator_rl.him.him_ppo:HIMPPO"
    hio_learning_rate: float = 3e-4
    hio_epochs: int = 1
    hio_batch_size: int = 128


@configclass
class HIMPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    init_at_random_ep_len = False
    experiment_name = "grallator_him_standing"
    run_name = "him_middle_to_height_v1"

    obs_groups = {
        "actor": ["policy"],
        "critic": ["critic"],
    }

    actor = HIMActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.3,
        ),
    )

    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )

    algorithm = HIMAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
