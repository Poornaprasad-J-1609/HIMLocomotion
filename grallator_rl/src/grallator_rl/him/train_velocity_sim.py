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
cfg.scene.num_envs = 32
cfg.seed = 42
# HIM starts from MIDDLE_POSE height
cfg.middle_base_height = 0.428

with launch_simulation(cfg.sim, args):

    env = StandingEnv(cfg)

    try:
        obs, _ = env.reset()

        # Check that we really start from MIDDLE_POSE
        start_height = env.robot.data.root_pos_w.torch[0, 2].item()

        pose_error = (
            env.robot.data.joint_pos.torch
            - env.middle_joint_pos
        ).abs().max().item()

        print("Initial base height:", start_height)
        print("Maximum initial joint error:", pose_error)

        if abs(start_height - 0.426) > 0.03 or pose_error > 0.04:
            raise RuntimeError(
                "Robot is not resetting near MIDDLE_POSE. "
                "Check robot.py and env.py before training."
            )

        assert obs["policy"].shape == (32, 45)

        # 45D observations, six frames each
        history = HIMHistory(
            num_envs=32,
            obs_dim=45,
            history_steps=6,
            device=env.device,
        )

        # Move HIM to the same GPU as Newton
        estimator = HIMEstimator().to(env.device)

        trainer = HIMTrainer(
            estimator,
            learning_rate=3e-4,
        )

        history.push(obs["policy"])

        actions = torch.zeros(32, 12, device=env.device)

        for step in range(400):

            # Small exploratory joint commands held for 10 steps
            if step % 10 == 0:
                actions.uniform_(-0.6, 0.6)

            obs, reward, terminated, truncated, _ = env.step(
                actions
            )

            done = terminated | truncated

            # Clear history from completed episodes
            if done.any():
                history.reset(torch.where(done)[0])

            # New observation and corresponding ground-truth velocity
            obs_history = history.push(obs["policy"])

            true_velocity = (
                env.robot.data.root_lin_vel_b.torch
                .detach()
                .clone()
            )

            assert torch.isfinite(obs_history).all()
            assert torch.isfinite(true_velocity).all()

            # Train on robots 0–23
            valid_train = ~done[:24]

            if valid_train.any():
                loss = trainer.update_velocity(
                    obs_history[:24][valid_train],
                    true_velocity[:24][valid_train],
                )

            # Evaluate on robots 24–31
            if step % 50 == 0:

                valid_eval = ~done[24:]

                if valid_eval.any():
                    with torch.no_grad():

                        predicted, _ = estimator(
                            obs_history[24:][valid_eval]
                        )

                        target = true_velocity[24:][valid_eval]

                        velocity_rmse = torch.sqrt(
                            torch.mean((predicted - target) ** 2)
                        )

                        # Error of always predicting zero velocity
                        zero_baseline = torch.sqrt(
                            torch.mean(target ** 2)
                        )

                        mean_speed = torch.linalg.vector_norm(
                            target, dim=1
                        ).mean()

                    print(
                        f"Step {step:03d} | "
                        f"Train MSE={loss:.6f} | "
                        f"Eval RMSE={velocity_rmse.item():.4f} m/s | "
                        f"Zero baseline={zero_baseline.item():.4f} m/s | "
                        f"Mean speed={mean_speed.item():.4f} m/s"
                    )

        # Save the standalone estimator experiment
        output = (
            Path(__file__).resolve().parents[3]
            / "checkpoints"
            / "him_velocity_smoke.pt"
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

        print("\nHIM NEWTON VELOCITY TEST COMPLETED")
        print("Checkpoint:", output)

    finally:
        env.close()