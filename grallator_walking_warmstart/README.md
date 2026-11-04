# Grallator standing-to-walking warm start

This adapter changes no Grallator source and never writes over the source checkpoint. It creates a new compatible checkpoint from `model_1700.pt`, retaining actor, critic, HIM estimator, and normalizer state but changing `distribution.std_param` from ~0.01955 to 0.12, clearing Adam optimizer moments, setting PPO learning rate to 1e-4, and resetting `iter` to zero. It assumes the user's inspected checkpoint has `distribution.std_param` with shape [12] and the previously observed RSL-RL checkpoint keys.

Run from `~/RL/Quadruped/IsaacLab` with the existing Python environment and a trusted checkpoint:

```bash
SOURCE="$HOME/RL/Quadruped/IsaacLab/logs/rsl_rl/grallator_him_standing/2026-11-03_21-06-02_middle_pose_038_resume/model_1700.pt"
OUTPUT="$HOME/RL/Quadruped/IsaacLabEureka/eureka_grallator/candidates/walking_init_std012.pt"
./.venv/bin/python ../IsaacLabEureka/eureka_grallator/prepare_warmstart.py --source "$SOURCE" --output "$OUTPUT" --std 0.12 --lr 1e-4
```

**Before training, ensure your new walking task registration and HIO freeze patch are already installed and `GRALLATOR_EUREKA_REWARD_FILE` points to your working flat baseline JSON.** Use a 30-update smoke test with `GRALLATOR_FREEZE_HIO=1`, then assess actual speed tracking, falls, PPO KL, and reset behavior before a longer run. `--checkpoint` is used to load the new checkpoint. The new `iter=0` ensures training starts from the beginning of the new run, although exact loader behavior is dependent on your local RSL-RL version.

This is weight transfer rather than an exact optimizer resume. The old critic weights and normalizer stay intact and may initially be mismatched with the walking reward scale. Keep the original standing checkpoint for comparison. Changing reward-search coefficients isn't necessary in the first smoke test, because you should isolate the effect of restored exploration.

`torch.load(weights_only=False)` must only be used with a trusted checkpoint, since pickled checkpoints can execute code.
