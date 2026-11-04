"""Convenient single-height Grallator HIM+PPO training launcher.

Usage (from ~/RL/Quadruped/IsaacLab):
    PYTHONPATH="$HOME/RL/Quadruped:$HOME/RL/Quadruped/grallator_rl/src" \
      uv run python ../grallator_rl/scripts/train_standing.py \
      --height 0.37 --iterations 25 --num-envs 32 --viz newton_gl

This script sets GRALLATOR_TARGET_HEIGHT BEFORE importing the environment cfg.
The height is a single command, never sampled from a range.
"""

import argparse
import math
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, required=True, help="One desired base height in metres")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--run-name", default="him_fixed_height")
    parser.add_argument("--viz", default=None, help="Optional Newton viewer, e.g. newton_gl")
    parser.add_argument("--checkpoint", default=None, help="Only compatible 271D/92D PPO checkpoints")
    args = parser.parse_args()

    if not math.isfinite(args.height) or args.height <= 0:
        parser.error("--height must be a positive, finite value in metres")
    if args.iterations < 1 or args.num_envs < 1:
        parser.error("--iterations and --num-envs must be positive")

    # The registered config reads this value when it is first imported.
    os.environ["GRALLATOR_TARGET_HEIGHT"] = str(args.height)

    import grallator_rl.him.registration  # noqa: F401
    from isaaclab_rl.entrypoints import run_train_cli

    argv = [
        "--rl_library", "rsl_rl",
        "--task", "GrallatorRl-HIM-Standing-Direct",
        "--num_envs", str(args.num_envs),
        "--max_iterations", str(args.iterations),
        "--run_name", args.run_name,
    ]
    if args.viz:
        argv += ["--viz", args.viz]
    if args.checkpoint:
        argv += ["--checkpoint", args.checkpoint]
    print(
        f"Training Grallator: reset MIDDLE_POSE at 0.428 m, "
        f"target height={args.height:.3f} m, "
        f"{args.num_envs} environments, {args.iterations} iterations"
    )
    raise SystemExit(run_train_cli(argv))


if __name__ == "__main__":
    main()
