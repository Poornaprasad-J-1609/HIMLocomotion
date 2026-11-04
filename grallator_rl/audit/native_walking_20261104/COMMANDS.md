# Reproduce and run native walking

All commands below use the existing environment and tested launcher flags. Run each section separately. The 25- and 300-iteration runs are supplied for you; this audit executed only short 1–3 iteration training runs.

## A. Setup and verified standing-derived warm-start

```bash
cd /home/poornaprasad/RL/Quadruped/IsaacLab
export PYTHONPATH=/home/poornaprasad/RL/Quadruped:/home/poornaprasad/RL/Quadruped/grallator_rl/src
export WALKING_INIT=/home/poornaprasad/RL/Quadruped/grallator_rl/checkpoints/walking_from_standing.pt
export GRALLATOR_FREEZE_HIO=0
export GRALLATOR_HIO_LR_OVERRIDE=1e-5
.venv/bin/python -c 'from grallator_rl.him.walking_checkpoint import verify_warm_start; print(verify_warm_start())'
```

## B. 25-iteration graphical Newton smoke, eight environments

```bash
.venv/bin/python ../grallator_rl/scripts/train_walking.py \
  --checkpoint "$WALKING_INIT" --height 0.38 \
  --num-envs 8 --iterations 25 --viz newton_gl \
  --run-name walking_viewer25
```

Check tracking error, contacts, falls and both KL tags. A standing policy need not learn useful walking in 25 iterations. With 24 rollout steps/iteration, 25 iterations span a full 599-step timeout unless an environment falls earlier. Episode airtime tags should therefore appear, including zero values.

## C. Headless experiment, only after B passes

```bash
.venv/bin/python ../grallator_rl/scripts/train_walking.py \
  --checkpoint "$WALKING_INIT" --height 0.38 \
  --num-envs 64 --iterations 300 --run-name walking_headless300
```

This starts a new walking experiment from the standing warm-start. To continue a saved **walking** experiment instead, use `--resume --checkpoint /absolute/path/to/model_N.pt`; resume was tested with optimizer state, iteration and frozen HIO. Set `GRALLATOR_FREEZE_HIO=1` to freeze HIO, or leave it `0` for active HIO.

## D. TensorBoard at http://127.0.0.1:6006

```bash
.venv/bin/tensorboard \
  --logdir_spec native:/home/poornaprasad/RL/Quadruped/IsaacLab/logs/rsl_rl/grallator_him_walking_native,audit:/home/poornaprasad/RL/Quadruped/grallator_rl/audit/native_walking_20261104/controlled_events_16 \
  --host 127.0.0.1 --port 6006
```

Look for `Reward_Rate/*`, `Episode_Reward/*`, `Walking/*` and native `Loss/*`. `Loss/ppo_kl`, `Loss/post_hio_kl`, `Loss/hio_velocity_mse` and `Loss/hio_swap_loss` retain their native names. The audit run supplies controlled completed-episode event proof; new native training supplies natural episode logs.

## E. Change reward weights only in reward_cfg.py

```bash
nano /home/poornaprasad/RL/Quadruped/grallator_rl/src/grallator_rl/tasks/walking_direct/reward_cfg.py
.venv/bin/python -c 'from grallator_rl.tasks.walking_direct.env_cfg import WalkingEnvCfg; print(WalkingEnvCfg().rewards)'
```

For example edit `feet_air_time: float = 2.0` or enable `stance_slip` with a small nonnegative coefficient. The reward code applies penalty signs. Changes affect the next process, not an already running experiment. Unsupported torque/power weights must remain zero until a validated effort signal exists; the config reports the reason if enabled. Keep baseline weights for reproducing this audit's regression tests.

## F. Tests, event read-back and report

```bash
.venv/bin/python ../grallator_rl/scripts/run_walking_tests.py
.venv/bin/python ../grallator_rl/scripts/audit_walking_runtime.py \
  --num-envs 16 \
  --output /home/poornaprasad/RL/Quadruped/grallator_rl/audit/native_walking_20261104/runtime_recheck.json
.venv/bin/python ../grallator_rl/scripts/inspect_walking_run.py \
  --log-dir /home/poornaprasad/RL/Quadruped/IsaacLab/logs/rsl_rl/grallator_him_walking_native/2026-11-04_12-44-44_audit_final16
cat /home/poornaprasad/RL/Quadruped/grallator_rl/audit/native_walking_20261104/REPORT.md
sha256sum /home/poornaprasad/RL/Quadruped/IsaacLab/logs/rsl_rl/grallator_him_standing/2026-11-03_21-06-02_middle_pose_038_resume/model_1700.pt
```

`run_walking_tests.py` includes PPO/HIO active/frozen updates, non-overlapping optimizer groups, estimator gradients, strict checkpoint loading, numeric GAE/timeout bootstrap and stale-gradient regression. It needs no GPU. Add `--all` to reproduce the full 25-test suite, including the three documented pre-existing standing fixture errors.

To independently load the trusted original with its unchanged standing task/launcher (training invocation, zero walking dependencies):

```bash
.venv/bin/python ../grallator_rl/scripts/train_standing.py \
  --height 0.38 --num-envs 8 --iterations 1 --viz newton_gl \
  --run-name standing_original_check \
  --checkpoint /home/poornaprasad/RL/Quadruped/IsaacLab/logs/rsl_rl/grallator_him_standing/2026-11-03_21-06-02_middle_pose_038_resume/model_1700.pt
```

That standing command uses the existing CLI but was not run in this audit, to avoid creating new standing logs. The original standing files/checkpoint remain unchanged.
