#!/usr/bin/env python3
"""Make a NEW walking fine-tuning checkpoint from a trusted standing checkpoint.

Preserve actor, critic, estimator and normalization weights. Reinitialize only
Gaussian exploration, PPO/HIO optimizer moments, and iteration counter.
This retains the checkpoint schema expected by the existing RSL-RL loader.
Only run with checkpoints produced by you or another trusted source:
torch.load(weights_only=False) can execute arbitrary Python code.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import torch


def prepare(source: Path, output: Path, std: float, lr: float) -> None:
    if not math.isfinite(std) or not (0.02 <= std <= 0.5):
        raise ValueError("std must be finite and in [0.02, 0.5]")
    if not math.isfinite(lr) or not (1e-6 <= lr <= 5e-4):
        raise ValueError("lr must be finite and in [1e-6, 5e-4]")

    source = source.expanduser().resolve(strict=True)
    output = output.expanduser().resolve()
    if source == output:
        raise ValueError("Refusing to overwrite the original checkpoint")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise TypeError("Expected checkpoint dictionary")
    required = ("actor_state_dict", "critic_state_dict", "optimizer_state_dict", "iter")
    missing = [k for k in required if k not in ckpt]
    if missing:
        raise KeyError(f"Missing checkpoint keys: {missing}")

    actor = ckpt["actor_state_dict"]
    std_key = "distribution.std_param"
    if std_key not in actor:
        raise KeyError(f"Expected {std_key}; verify action-distribution implementation")
    original_std = actor[std_key]
    if not torch.is_tensor(original_std) or original_std.shape != (12,):
        raise ValueError("Expected a tensor of 12 Gaussian standard deviations")
    if not torch.isfinite(original_std).all().item() or not (original_std > 0).all().item():
        raise ValueError("Checkpoint std values must be finite and positive")
    previous_std = original_std.float().mean().item()
    actor[std_key] = torch.full_like(original_std, std)

    ppo_optimizer = ckpt["optimizer_state_dict"]
    if not isinstance(ppo_optimizer, dict) or not ppo_optimizer.get("param_groups"):
        raise TypeError("Unexpected PPO optimizer state_dict layout")
    ppo_optimizer["state"] = {}  # Reset Adam momentum and variance estimates.
    for group in ppo_optimizer["param_groups"]:
        group["lr"] = lr
        if "initial_lr" in group:
            group["initial_lr"] = lr

    if "hio_optimizer_state_dict" in ckpt:
        hio_optimizer = ckpt["hio_optimizer_state_dict"]
        if not isinstance(hio_optimizer, dict) or not hio_optimizer.get("param_groups"):
            raise TypeError("Unexpected HIO optimizer state_dict layout")
        hio_optimizer["state"] = {}  # HIO stays frozen in this first experiment.

    previous_iter = ckpt["iter"]
    ckpt["iter"] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    # xb: don't silently destroy an existing checkpoint.
    with output.open("xb") as f:
        torch.save(ckpt, f)

    print(f"SOURCE: {source}")
    print(f"OUTPUT: {output}")
    print(f"Iteration: {previous_iter} -> 0")
    print(f"Actor action std: {previous_std:.6f} -> {std:.6f}")
    print(f"PPO LR: {lr:.6g}; PPO/HIO Adam moments cleared")
    print("Actor/critic/estimator weights unchanged; source file untouched")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--std", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    prepare(args.source, args.output, args.std, args.lr)


if __name__ == "__main__":
    main()
