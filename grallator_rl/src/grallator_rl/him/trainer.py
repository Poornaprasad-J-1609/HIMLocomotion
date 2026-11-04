import torch
import torch.nn.functional as F
from grallator_rl.him.contrastive import compute_hio_losses


class HIMTrainer:

    def __init__(self, estimator, learning_rate=1e-3):
        self.estimator = estimator

        self.optimizer = torch.optim.Adam(
            self.estimator.parameters(),
            lr=learning_rate,
        )

    def update_velocity(self, history, true_velocity):

        # Validate input dimensions
        assert history.ndim == 2
        assert history.shape[1] == 270
        assert true_velocity.shape == (history.shape[0], 3)

        # HIM predicts velocity from recent observations
        predicted_velocity, latent = self.estimator.encode(
            history.detach()
        )

        # Newton provides supervised velocity labels
        target_velocity = true_velocity.detach()

        # Mean squared estimation error
        velocity_loss = F.mse_loss(
            predicted_velocity,
            target_velocity,
        )

        # Backpropagation
        self.optimizer.zero_grad(set_to_none=True)
        velocity_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.estimator.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        return velocity_loss.detach().item()

    def update_hio(
        self,
        history_t,
        next_obs,
        next_true_velocity,
    ):

        velocity_loss, swap_loss = compute_hio_losses(
            self.estimator,
            history_t,
            next_obs,
            next_true_velocity,
        )

        # Train the source encoder, target encoder and prototypes
        total_loss = velocity_loss + swap_loss

        self.optimizer.zero_grad(set_to_none=True)

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.estimator.parameters(),
            max_norm=10.0,
        )

        self.optimizer.step()

        return (
            velocity_loss.detach().item(),
            swap_loss.detach().item(),
        )