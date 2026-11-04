# Grallator — Fixed-Height Standing with HiMLoco + PPO

This is the cleaned **Grallator 12-DOF** standing-training project for **Isaac Lab 3.0 Kitless / Newton**, RSL-RL PPO and a separately updated HiMLoco HIO estimator. It is a refactor of the uploaded project, **not a simulator-validated pretrained standing policy**.

## What the task does

**Every reset:** Put all 12 joints at the exact `MIDDLE_POSE`, put the base at the configured middle-pose reset height, and set linear/angular velocities to zero. **Every step:** command a joint target relative to `MIDDLE_POSE`, and reward maintaining **one specified base height** while staying upright, symmetrical, quiet, and keeping only feet in contact.

- `middle_base_height = 0.428 m` (reset root z).
- `target_base_height = 0.50 m` (kept from the uploaded archive; **unverified feasibility**).
- **No** `height_command_min`, `height_command_max`, or randomized height command.
- Changing the requested height changes the reward target *and* the height feature seen by PPO. A fixed-height policy trained on one height is **not guaranteed** to work at another: retrain or fine-tune when changing it.
- `MIDDLE_POSE`:

  | Leg | Hip | Thigh | Calf |
  |---|---:|---:|---:|
  | FR | 0.000 | -0.325 | -0.675 |
  | FL | 0.000 | +0.325 | +0.675 |
  | BR | 0.000 | -0.325 | -0.675 |
  | BL | 0.000 | +0.325 | +0.675 |

  (Angles in radians; virtual calf joint, **not** the physical four-bar motor-crank angle.)

- PPO commands `q_target = MIDDLE_POSE + action_scales * clipped_action`, with hip scale 0.06 rad and thigh/calf scale 0.25 rad, then respects imported joint position limits. These are simulation starting points, not verified physical-hardware safe ranges.

## Where everything belongs

```text
grallator_rl/
├── checkpoints/                             # Existing HIM pretraining checkpoints
├── scripts/
│   ├── train_standing.py                    # One-height training launcher
│   └── list_envs.py
├── src/grallator_rl/
│   ├── tasks/standing_direct/
│   │   ├── rewards.py                       # Pure PyTorch standing reward math
│   │   └── config/grallator/
│   │       ├── env_cfg.py                   # Fixed height, MIDDLE_POSE, scene, action scales
│   │       ├── reward_cfg.py                # All reward coefficients and tolerances
│   │       └── env.py                       # Reset, actions, observations, contacts, logging
│   └── him/
│       ├── env_adapter.py                   # 45D x 6 history + height; 91D critic + height
│       ├── history.py, estimator.py         # HiMLoco estimator (unchanged 270D interface)
│       ├── trainer.py, contrastive.py       # HIO training
│       ├── rsl_actor.py, him_ppo.py          # RSL-RL integration
│       └── him_ppo_cfg.py, registration.py  # PPO configuration, task registration
└── tests/
    ├── test_standing_math.py               # CPU-only reward checks
    └── test_him_shapes.py                  # CPU-only HIM shape checks
```

`him/actor.py`, `him/critic.py`, `him/train_velocity_sim.py`, and `him/train_him_sim.py` are retained as helper/experimental code; the PPO training path uses the custom `rsl_actor.py` and `him_ppo.py` implementations.

## Dependencies

This is a **project extension** and does not include Isaac Lab or the robot asset. It requires:

- Isaac Lab 3.0 Kitless with Isaac Lab Newton, the `rsl_rl` library, and Python 3.12, configured as in `pyproject.toml`.
- Your separate `grallator` Python package supplying `grallator.config.robot.GRALLATOR_CFG` (the USD/URDF articulation and its collision geometries) and `grallator.config.observations.build_policy_observation` (45D convention).
- The robot USD must include valid collision geometries for all relevant links, since no collision geometry means no contact forces or penalties.
- The default `checkpoints/him_hio_smoke.pt` is preserved from the upload.

## Train at one height

From `~/RL/Quadruped/IsaacLab`:

```bash
cd ~/RL/Quadruped/IsaacLab

PYTHONPATH="$HOME/RL/Quadruped:$HOME/RL/Quadruped/grallator_rl/src" \
uv run python ../grallator_rl/scripts/train_standing.py \
  --height 0.37 --num-envs 32 --iterations 25 \
  --run-name middle_fixed_037_smoke --viz newton_gl
```

After successful smoke testing, you can request more iterations (same target height):

```bash
PYTHONPATH="$HOME/RL/Quadruped:$HOME/RL/Quadruped/grallator_rl/src" \
uv run python ../grallator_rl/scripts/train_standing.py \
  --height 0.37 --num-envs 32 --iterations 300 \
  --run-name middle_fixed_037_training --viz newton_gl
```

That **second command starts a new PPO run** unless you supply `--checkpoint /absolute/path/to/model_N.pt` from a **checkpoint produced with this cleaned package**. The RSL-RL runner saves a final checkpoint at normal completion. Historical PPO checkpoint optimizer states from the pre-refactor package may be incompatible because the PPO and HIM optimizers are now separated. The configuration still loads the pretrained HIM estimator but does not load an old full PPO model unless requested.

For visual training, add `--viz newton_gl`. Omit it for faster headless training. TensorBoard:

```bash
cd ~/RL/Quadruped/IsaacLab
uv run tensorboard --logdir logs/rsl_rl/grallator_him_standing --port 6006
```

The target can also be set by `GRALLATOR_TARGET_HEIGHT=0.37` with the existing Isaac Lab runner CLI, or by editing `target_base_height` in `env_cfg.py`. The new launcher handles the environment-variable timing for you.

## Rewards and penalties

`StandingEnv._get_rewards()` constructs named reward terms through pure `rewards.py`, scales them by `self.step_dt` **once** and logs episode contributions. All coefficients are in `reward_cfg.py`:

- **Positive:** `upright`, `height` (accurate target), `height_progress` (dense shaping for getting closer).
- **Negative:** `angular_velocity`, `yaw_velocity`, `base_lin_vel`, `vertical_velocity`, `joint_velocity`, `pose_deviation` (small thigh/calf-only penalty), `leg_symmetry`, `foot_height_balance`, `action_rate`, `action_magnitude`, `hip_stability`, `undesired_contact`.
- `pose_deviation` is deliberately weak **and excludes the hips**: PPO must be free to flex its thighs/calves away from MIDDLE_POSE to reach the user-provided height. A separate `hip_stability` penalty favors hips near their initial zeros.
- `undesired_contact`: non-foot contact force magnitude greater than **2 N** incurs a cost, once per physics-control step. **It uses aggregate contacts:** it can penalize self-collisions or obstacles, not just ground. Check sensor body names printed at startup and verify collision meshes.
- Fall termination when the body is below **0.09 m** or markedly tipped. This is a failure/reset condition, **not** a min/max *commanded target height*.

## What to validate before long training

1. In Newton viewer verify the robot **spawns in MIDDLE_POSE** at 0.428 m, with feet not penetrating the plane; check collision geometry and root/base measurement conventions.
2. Check target height is reachable with real joint limits and plausible actuator torques, especially if using the **unverified 0.50 m** default.
3. Compare `Diagnostics/base_height_m`, `Diagnostics/height_target_m`, `Diagnostics/height_error_cm`, `Diagnostics/symmetry_rmse_deg`, `Diagnostics/undesired_contact_pct`, `Diagnostics/action_near_limit_pct` and `Metrics/fall_fraction` in TensorBoard; KL metrics are logged by the custom PPO algorithm.
4. If all contacts register zero even when a non-foot link visibly touches the floor, check collision geometry, the printed matched sensor bodies, and sensor activation. Do not treat zero force alone as proof of no contact.
5. Verify the deterministic PPO policy separately. Training explores with Gaussian action sampling; rewards alone do not establish stable standing.

## Tests and scope

The project was statically compiled and CPU-only reward/HIM tests were run when packaging. **No Isaac Lab / Newton simulation was available in the packaging environment**, so successful real simulation, exact foot clearance, dynamics, contact detection, and training convergence are **not yet verified**.

From project root (in a Python environment with torch, pytest):

```bash
PYTHONPATH=src pytest -q tests/test_standing_math.py tests/test_him_shapes.py
python -m compileall -q src scripts tests
```

See `CHANGELOG_FIXED_HEIGHT.md` for a detailed change list.
