"""Walking-only gradient hygiene around the unchanged shared HIM/PPO update."""
from .him_ppo import HIMPPO


class WalkingPPO(HIMPPO):
    def update(self):
        # HIO leaves estimator gradients after its optimizer step. Estimator
        # parameters are excluded from the PPO optimizer, so PPO.zero_grad()
        # cannot clear them. Native PPO clips *all* actor parameters: stale HIO
        # gradients would otherwise affect the next actor clipping norm even
        # though get_latent() correctly detaches the estimator from PPO.
        self._raw_actor.estimator.zero_grad(set_to_none=True)
        return super().update()
