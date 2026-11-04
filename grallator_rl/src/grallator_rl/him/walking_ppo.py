"""Walking-only gradient hygiene around the unchanged shared HIM/PPO update."""
from .him_ppo import HIMPPO


class WalkingPPO(HIMPPO):
    @staticmethod
    def construct_algorithm(obs, env, cfg, device):
        algorithm = HIMPPO.construct_algorithm(obs, env, cfg, device)
        algorithm._torque_env = env.unwrapped
        algorithm._torque_env.diagnostics.torque.accumulator.reset_run()
        return algorithm

    def update(self):
        # HIO leaves estimator gradients after its optimizer step. Estimator
        # parameters are excluded from the PPO optimizer, so PPO.zero_grad()
        # cannot clear them. Native PPO clips *all* actor parameters: stale HIO
        # gradients would otherwise affect the next actor clipping norm even
        # though get_latent() correctly detaches the estimator from PPO.
        self._raw_actor.estimator.zero_grad(set_to_none=True)
        metrics = super().update()
        if hasattr(self, '_torque_env'):
            self._torque_env.diagnostics.publish_torque_iteration(self._torque_env.extras['log'])
        return metrics
