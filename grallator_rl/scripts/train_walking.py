"""Native Grallator walking: original Newton + custom HIM/PPO.

Rewards: grallator_rl/tasks/walking_direct/reward_cfg.py and rewards.py
Commands: grallator_rl/tasks/walking_direct/env_cfg.py
No LLM, JSON reward file, or automated reward search.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--height', type=float, default=0.38)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--num-envs', type=int, default=64)
    parser.add_argument('--run-name', default='him_native_walking')
    parser.add_argument('--viz', default=None)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--resume', action='store_true', help='Resume a walking run including optimizer and iteration state')
    args = parser.parse_args()
    if not math.isfinite(args.height) or args.height <= 0:
        parser.error('--height must be positive and finite')
    if args.iterations < 1 or args.num_envs < 1:
        parser.error('--iterations and --num-envs must be positive')
    from grallator_rl.him.walking_checkpoint import WARM_START, verify_warm_start
    checkpoint = Path(args.checkpoint or WARM_START).expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        parser.error('--checkpoint must point to a file')
    if args.resume and not args.checkpoint:
        parser.error('--resume requires an explicit walking --checkpoint')
    if not args.resume:
        print('Verified standing provenance:', verify_warm_start(checkpoint), flush=True)
    os.environ.setdefault('GRALLATOR_FREEZE_HIO', '0')
    os.environ.setdefault('GRALLATOR_HIO_LR_OVERRIDE', '1e-5')
    os.environ['GRALLATOR_TARGET_HEIGHT'] = str(args.height)
    import grallator_rl.him.walking_registration  # noqa: F401
    from isaaclab_rl.entrypoints import run_train_cli
    argv = [
        '--rl_library', 'rsl_rl',
        '--task', 'GrallatorRl-HIM-Walking-Direct',
        '--num_envs', str(args.num_envs),
        '--max_iterations', str(args.iterations),
        '--run_name', args.run_name,
    ]
    if checkpoint:
        argv += ['--checkpoint', str(checkpoint)]
    if not args.resume:
        argv += ['--no-resume_optimizer_state']
    if args.viz:
        argv += ['--viz', args.viz]
    print(f'Native walking: height={args.height:.3f} m, envs={args.num_envs}, iterations={args.iterations}')
    print(f'Starting checkpoint: {checkpoint if checkpoint else "none"}')
    print('Rewards: grallator_rl/tasks/walking_direct/reward_cfg.py')
    raise SystemExit(run_train_cli(argv))


if __name__ == '__main__':
    main()
