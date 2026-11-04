import torch
import torch.nn as nn
import torch.nn.functional as F

class HIMEstimator(nn.Module):

    def __init__(
        self,
        obs_dim=45,
        history_steps=6,
        latent_dim=16,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.history_steps = history_steps
        self.latent_dim = latent_dim

        input_dim = obs_dim * history_steps
        output_dim = 3 + latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, output_dim),
        )
        # Target encoder for contrastive learning
        self.target = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, latent_dim),
        )

        # 32 learnable prototypes
        self.prototypes = nn.Embedding(32, latent_dim)

        self.temperature = 3.0

    def encode(self, obs_history):

        if (
            obs_history.ndim != 2
            or obs_history.shape[1]
            != self.obs_dim * self.history_steps
        ):
            raise ValueError(
                f"Expected history [N, "
                f"{self.obs_dim * self.history_steps}], "
                f"got {tuple(obs_history.shape)}"
            )

        output = self.encoder(obs_history)

        # First three outputs: estimated linear velocity
        velocity = output[:, :3]

        # Remaining outputs: learned latent representation
        latent = output[:, 3:]

        # Normalize the latent to unit length
        latent = F.normalize(
            latent,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        return velocity, latent

    def forward(self, obs_history):
        # Use frozen estimator outputs for actor inference.
        # Estimator training will call encode() directly.
        with torch.no_grad():
            return self.encode(obs_history)