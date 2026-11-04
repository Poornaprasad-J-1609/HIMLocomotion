"""Train or fine-tune Grallator walking with Newton and the original HIM/PPO.

EXAMPLE:
  uv run python ../grallator_rl/scripts/train_walking.py \
    --height 0.40 --iterations 10 --num-envs 64 \
    --checkpoint /ABSOLUTE/PATH/model_1500.pt --run-name walk_smoke

Existing standing task, reward_cfg.py, environment, and checkpoints are untouched.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=float, default=0.40)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--run-name", default="him_flat_walking_smoke")
    parser.add_argument("--viz", default=None)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    if not math.isfinite(args.height) or args.height <= 0:
        parser.error("--height must be positive and finite")
    if args.iterations < 1 or args.num_envs < 1:
        parser.error("--iterations and --num-envs must be positive")
    if args.checkpoint is not None:
        checkpoint = Path(args.checkpoint).expanduser().resolve(strict=True)
        if not checkpoint.is_file():
            parser.error("--checkpoint must reference a file")
    else:
        checkpoint = None
    os.environ["GRALLATOR_TARGET_HEIGHT"] = str(args.height)
    import grallator_rl.him.walking_registration  # noqa: F401
    from isaaclab_rl.entrypoints import run_train_cli
    argv = [
        "--rl_library", "rsl_rl",
        "--task", "GrallatorRl-HIM-Walking-Direct",
        "--num_envs", str(args.num_envs),
        "--max_iterations", str(args.iterations),
        "--run_name", args.run_name,
    ]
    if checkpoint:
        argv += ["--checkpoint", str(checkpoint)]
    if args.viz:
        argv += ["--viz", args.viz]
    print(f"Walking: height={args.height:.3f}m, envs={args.num_envs}, iters={args.iterations}")
    print(f"Checkpoint: {checkpoint or '<none>'}")
    print("Walking reward config:", os.environ.get("GRALLATOR_WALK_REWARD_FILE", "walking_direct/reward_weights.json"))
    raise SystemExit(run_train_cli(argv))


if __name__ == "__main__":
    main()
