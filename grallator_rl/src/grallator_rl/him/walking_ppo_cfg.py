"""Walking-only logging and initial HIO settings; standing config is unchanged."""
from isaaclab.utils import configclass
from .him_ppo_cfg import HIMPPORunnerCfg


@configclass
class WalkingPPORunnerCfg(HIMPPORunnerCfg):
    experiment_name = 'grallator_him_walking_native'
    run_name = 'native_walking'

    def __post_init__(self):
        self.algorithm.class_name = 'grallator_rl.him.walking_ppo:WalkingPPO'
        self.algorithm.hio_learning_rate = 1e-5
        # The standing warm-start's first update at 3e-4 produced KL ~8 in
        # the 16-env audit. Start conservatively; retain native adaptive PPO.
        self.algorithm.learning_rate = 1e-5
        # The full actor (including estimator) is loaded from the required
        # standing-derived checkpoint, so a second estimator file is unnecessary.
        self.actor.estimator_checkpoint_path = None
