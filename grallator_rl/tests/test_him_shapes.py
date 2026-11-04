"""CPU-only checks that the HIM encoder still trains on 270D histories."""

from pathlib import Path
import math

import torch

from grallator_rl.him.estimator import HIMEstimator
from grallator_rl.him.trainer import HIMTrainer


def test_velocity_trainer_uses_pretrained_270d_interface():
    torch.manual_seed(42)
    estimator = HIMEstimator(obs_dim=45, history_steps=6, latent_dim=16)
    trainer = HIMTrainer(estimator, learning_rate=1e-4)
    obs = torch.randn(8, 270)
    vel = torch.randn(8, 3) * 0.1
    mse = trainer.update_velocity(obs, vel)
    assert math.isfinite(mse)
    v, z = estimator.encode(obs)
    assert v.shape == (8, 3)
    assert z.shape == (8, 16)


def test_hio_update_runs_with_270d_history_and_45d_next_obs():
    torch.manual_seed(42)
    estimator = HIMEstimator()
    trainer = HIMTrainer(estimator, learning_rate=1e-4)
    velocity_loss, swap_loss = trainer.update_hio(
        torch.randn(8, 270),
        torch.randn(8, 45),
        torch.randn(8, 3) * 0.1,
    )
    assert math.isfinite(velocity_loss)
    assert math.isfinite(swap_loss)


def test_existing_hio_checkpoint_matches_encoder_architecture():
    checkpoint_path = Path(__file__).resolve().parents[1] / "checkpoints/him_hio_smoke.pt"
    if not checkpoint_path.exists():
        return  # Checkpoint is optional for a source-only checkout.
    model = HIMEstimator()
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["estimator"], strict=True)
    velocity, z = model.encode(torch.zeros(2, 270))
    assert velocity.shape == (2, 3)
    assert z.shape == (2, 16)
