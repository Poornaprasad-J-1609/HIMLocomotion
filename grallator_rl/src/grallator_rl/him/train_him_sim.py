import argparse
from pathlib import Path

import torch

from isaaclab.app import add_launcher_args, launch_simulation

from grallator_rl.him.history import HIMHistory
from grallator_rl.him.estimator import HIMEstimator
from grallator_rl.him.trainer import HIMTrainer

from grallator_rl.tasks.standing_direct.config.grallator.env_cfg import (
    StandingEnvCfg,
)
from grallator_rl.tasks.standing_direct.config.grallator.env import (
    StandingEnv,
)


parser = argparse.ArgumentParser()
add_launcher_args(parser)
args = parser.parse_args()

cfg = StandingEnvCfg()
cfg.scene.num_envs = 64
cfg.seed = 42
cfg.target_base_height = 0.3631

# Compatibility with our earlier crouch-reset configuration.
# The actual joints must still reset near MIDDLE_POSE.
if hasattr(cfg, "middle_base_height"):
    cfg.middle_base_height = 0.428


with launch_simulation(cfg.sim, args):
    env = StandingEnv(cfg)

    try:
        obs, _ = env.reset()

        assert obs["policy"].shape == (64, 45)

        # Verify MIDDLE_POSE initialization.
        start_height = env.robot.data.root_pos_w.torch[0, 2].item()

        pose_error = (
            env.robot.data.joint_pos.torch
            - env.middle_joint_pos
        ).abs().max().item()

        print("Initial height:", start_height)
        print("Maximum initial joint error:", pose_error)

        if abs(start_height - 0.426) > 0.03 or pose_error > 0.04:
            raise RuntimeError(
                "Reset is not near MIDDLE_POSE. "
                "Remove remaining crouch joint overrides in env.py."
            )

        history = HIMHistory(
            num_envs=64,
            obs_dim=45,
            history_steps=6,
            device=env.device,
        )

        estimator = HIMEstimator(
            obs_dim=45,
            history_steps=6,
            latent_dim=16,
        ).to(env.device)

        # Initialize the source encoder from our velocity-only run.
        old_checkpoint = (
            Path(__file__).resolve().parents[3]
            / "checkpoints"
            / "him_velocity_smoke.pt"
        )

        if old_checkpoint.exists():
            checkpoint = torch.load(
                old_checkpoint,
                map_location=env.device,
                weights_only=True,
            )

            result = estimator.load_state_dict(
                checkpoint["estimator"],
                strict=False,
            )

            if result.unexpected_keys:
                raise RuntimeError(
                    f"Unexpected checkpoint keys: {result.unexpected_keys}"
                )

            print("Loaded previous velocity-estimator weights")
            print("Newly initialized parameters:", result.missing_keys)

        trainer = HIMTrainer(
            estimator,
            learning_rate=3e-4,
        )

        # H_t: history ending at the current observation.
        history_t = history.push(obs["policy"])

        actions = torch.zeros(64, 12, device=env.device)

        print("\n===== HIM HIO TRAINING =====")

        for step in range(800):

            # Introduce motion to make velocity estimation meaningful.
            if step % 10 == 0:
                actions.uniform_(-0.6, 0.6)

            # Advance Newton from t to t+1.
            next_obs, reward, terminated, truncated, _ = env.step(
                actions
            )

            done = terminated | truncated

            # Successor velocity label: v_(t+1).
            next_velocity = (
                env.robot.data.root_lin_vel_b.torch
                .detach()
                .clone()
            )

            assert torch.isfinite(next_obs["policy"]).all()
            assert torch.isfinite(next_velocity).all()

            # Train using environments 0 to 47.
            valid_train = ~done[:48]

            if valid_train.any():
                velocity_loss, swap_loss = trainer.update_hio(
                    history_t[:48][valid_train],
                    next_obs["policy"][:48][valid_train],
                    next_velocity[:48][valid_train],
                )

            # Evaluate using environments 48 to 63.
            if step % 50 == 0:
                valid_eval = ~done[48:]

                if valid_eval.any():
                    with torch.no_grad():
                        predicted_velocity, latent = estimator(
                            history_t[48:][valid_eval]
                        )

                        target_velocity = next_velocity[48:][valid_eval]

                        eval_rmse = torch.sqrt(
                            torch.mean(
                                (predicted_velocity - target_velocity) ** 2
                            )
                        )

                        zero_baseline = torch.sqrt(
                            torch.mean(target_velocity ** 2)
                        )

                    print(
                        f"Step {step:03d} | "
                        f"Velocity MSE={velocity_loss:.6f} | "
                        f"Swap loss={swap_loss:.6f} | "
                        f"Eval RMSE={eval_rmse.item():.4f} m/s | "
                        f"Zero baseline={zero_baseline.item():.4f} m/s"
                    )

            # Prevent histories from crossing episode boundaries.
            if done.any():
                history.reset(
                    done.nonzero(as_tuple=True)[0]
                )
                actions[done] = 0.0

            # Now build the history ending at t+1.
            history_t = history.push(next_obs["policy"])

        output = (
            Path(__file__).resolve().parents[3]
            / "checkpoints"
            / "him_hio_smoke.pt"
        )

        output.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "estimator": estimator.state_dict(),
                "obs_dim": 45,
                "history_steps": 6,
                "latent_dim": 16,
            },
            output,
        )

        print("\nHIM HIO NEWTON TRAINING COMPLETED")
        print("Saved checkpoint:", output)

    finally:
        env.close()