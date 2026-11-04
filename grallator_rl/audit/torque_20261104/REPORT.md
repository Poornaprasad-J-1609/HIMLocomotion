# Grallator per-actuator torque telemetry — 4 November 2026

The walking task now logs **solver-reported joint-side actuation torque** for all 12 joints from Newton MJWarp's `qfrc_actuator`. It samples after every **0.005 s physics step**, accumulates on the GPU, and publishes each statistic **once per PPO iteration through the existing native RSL-RL writer**. Approximate unclipped PD curves are separately labelled `EstimatedTorque/`.

No standing source, original checkpoint, robot geometry, actuator parameters, reward functions/coefficients, observation/action assembly, shared HIM/PPO mathematics or checkpoint schema were changed. Torque/power penalties remain disabled and their existing guards remain intact.

## Audit

| Requirement | Result | Executed evidence |
|---|---|---|
| Actual signal provenance | PASS | `signal_probe.log`: implicit drives, native explicit-actuator path inactive; public effort is approximate PD; backend `qfrc_actuator` exists. Controlled command produces signed, nonzero solver torque and observed saturation. |
| Solver force validation | PASS | `runtime_warmed.json`: all 12 hinge DOF torques agree with the sum of their two native P/D actuator contributions times gear, with the solver's joint-level force range applied. Actual compiled joint force-limit flags and ±100 N·m ranges match runtime articulation limits. |
| Exactly 12 named joints and order | PASS | CPU rejection tests cover missing/duplicate/extra names. Runtime resolves names independently in articulation and MuJoCo model; mapping below. Floating-base DOFs and 24 scalar P/D entries are excluded from the 12-joint reporting axis. |
| Positive/negative, multi-env/time peaks; mean/P95/fraction | PASS | `cpu_tests.log`: actual Warp kernel tested on CPU with known signed tensors. Exact P95 uses linear interpolation over all env/time samples. |
| Physics-step sampling and CUDA graph replay | PASS | Training event counts are exactly `24 × 4 × num_envs` samples **per joint per iteration**: 768 at 8 envs, 9,600 at 100 envs. Sample interval 0.005 s. No per-step host readback. |
| Interval and run boundaries | PASS | CPU tests: interval peak resets, maximum-ever persists, run reset clears it, old snapshots remain immutable. Real per-env episode reset preserves accumulated torque statistics. New process/resumed run starts a new maximum; telemetry is not added to checkpoints. |
| Finite values and overflow | PASS | CPU tests inject NaN and buffer overflow; validation fails at the iteration boundary. Real events and controlled solver values are finite. |
| Native TensorBoard flow | PASS | `events.json`: all **62 required** tags exist, plus 12 optional `EstimatedTorque` curves and sampling metadata. Each tag has exactly one value per iteration. Native logger test proves interval peak 9 stays 9 rather than becoming an average of step peaks. |
| Training/policy preservation | PASS | Same stored rollout data with monitoring on/off produces bitwise identical PPO/HIO losses, gradients, actor/critic tensors and checkpoint keys. Seven telemetry tests and 17 existing focused walking tests pass. |
| Newton training / viewer | PASS | Three headless iterations at 8 and 100 envs; two iterations at 8 envs with `--viz newton_gl`; finite losses and saved checkpoints. `train_smoke8.log`, `train_smoke100.log`, `train_viewer8.log`. |
| Overhead | PASS for short local test | Warmed 24-control-step batches: baseline 141.82 ms, monitored 139.98 ms; summary 0.762 ms. The −1.3% collection delta is timing noise, not a claimed speedup. No measurable collection slowdown in this test; summary adds about 0.54% of baseline collection time. Native scalar writer cost is additional and was exercised in the training runs. |
| Identical independent physics rollouts | NOT VERIFIED | Separate contact-solver runs were not bitwise identical. Their comparison is recorded in runtime JSON; it is not used to claim deterministic physics. The identical-input optimizer test above passed. |
| Physical motor shaft torque / hardware calibration | NOT VERIFIED | These are simulator hinge actuation torques, not motor current measurements, joint reaction loads, or four-bar motor shaft torque. No transmission Jacobian or efficiency model was invented. |
| Protected files/checkpoint | PASS | `preservation.json`; original SHA-256 below. Only three existing files changed, all for monitoring integration. |

## Installed stack and signal meanings

Versions from installed distribution metadata (`versions.json`): Isaac Lab **28.0.0**, Isaac Lab Newton **9.1.4**, Newton **1.6.1**, MuJoCo **3.12.0**, MuJoCo Warp **3.12.0**, Warp **1.17.0**, RSL-RL **5.5.1**, Torch **2.11.0+cu130**.

`grallator/config/actuators.py` configures one `ImplicitActuatorCfg` group for all 12 rotational joints. These are solver P/D drives, not explicit Newton actuator models. Kp=250, Kd=4 and the 100 N·m effort limit are unchanged. The actuator group passes targets to the solver and computes approximate public effort telemetry alongside them.

| Candidate | Meaning and units | Clipping | Frequency / shape / ordering |
|---|---|---|---|
| `robot.actuators.applied_effort` | Approximate `Kp*(q_target-q)+Kd*(qd_target-qd)+feedforward`, N·m for these hinges. **Not solver-measured torque.** | Public implicit-actuator estimate clipped to runtime effort limits. | Model evaluation before each 0.005 s physics step in current task; `[N,12]`, articulation user joint order. |
| `robot.actuators.computed_effort` | Same approximate PD calculation before clipping, N·m. **EstimatedTorque** optional graphs only. | Unclipped in this all-implicit, non-native-explicit path. | Same model evaluation interval, `[N,12]`, articulation order. |
| `robot.data.applied_torque` | Alias of `actuators.applied_effort`; runtime data-pointer equality verified. The name does not make it measured. | Same as public applied effort. | Same `[N,12]` and ordering; no independent measurement. |
| `robot.data.computed_torque` | Source-defined alias of `actuators.computed_effort`. | Same as computed effort. | Same `[N,12]`; alias confirmed in installed source. |
| Newton control `joint_f` / articulation `_sim_bind_joint_effort` | Feed-forward control input, not total implicit-drive torque. Walking supplies zero feed-forward effort. | Input semantics; not a validated total-force signal. | Articulation view `[N,12]`; backing Newton DOFs also include floating base. Updated with command submission. Not logged as experienced torque. |
| `NewtonManager._solver.mjw_data.qfrc_actuator` | **Solver-reported generalized actuation torque** after the solver evaluates P/D drives and joint actuator-force caps. Not an approximation recalculated by this module. Not hardware measurement or net constraint/contact reaction torque. | Compiled model `jnt_actfrclimited` and `jnt_actfrcrange` verified for each hinge. Current limits ±100 N·m. | Updated every solver step; post-step callback samples each 0.005 s physics step. `[N,18]`: six floating-base DOFs plus 12 hinge DOFs in MuJoCo order. Selected output `[N,12]` uses explicit name mapping. |
| `solver.mjw_data.actuator_force` | Native scalar P/D servo contributions. These **24 entries are not 12 physical motor measurements**. Used only for independent solver-force reconstruction. | Individual contributions are not a substitute for the joint-level sum cap. | `[N,24]`, MuJoCo actuator order. Combine moments and apply joint cap to compare with `qfrc_actuator`. |
| Newton state `state.mujoco.qfrc_actuator` | Optional converted state output exists in Newton source but is **not allocated in this task** (`None` in probe). | Not applicable. | Not used; no fictitious signal was substituted. |

Example actually observed in `signal_probe.log`: FR calf computed PD estimate **110.01297 N·m**, public clipped estimate **100 N·m**, solver output **100 N·m**. The controlled runtime audit observed solver peak **100 N·m** and an unclipped PD peak about **110.3919 N·m**. Logged peaks are calculated from sampled force values; the configured limit is used only to calculate the near-limit fraction, never inserted as a peak value.

Source evidence, relative to `/home/poornaprasad/RL/Quadruped`:

- `IsaacLab/source/isaaclab/isaaclab/actuators/actuator_pd.py:197`: implicit model explicitly documents approximate computed/applied telemetry and its PD formula.
- `IsaacLab/source/isaaclab/isaaclab/assets/articulation/base_articulation_data.py:80`: public torque aliases.
- `IsaacLab/source/isaaclab_newton/isaaclab_newton/assets/articulation/actuator_control.py:139`: submits implicit position/velocity drives and feed-forward inputs.
- `IsaacLab/.venv/lib/python3.12/site-packages/newton/_src/solvers/mujoco/solver_mujoco.py:7027`: joint-level actuator force cap; `:5293`: solver actuator-force output.
- `IsaacLab/source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py:1933`: graph-safe post-step callback API. Current manager decimation is 1 and solver substeps 1; environment control decimation is 4.

## Joint mapping

The reporting order is FR, FL, BR, BL; hip, thigh, calf within each leg. Names are checked for exact membership/uniqueness. Each solver match must be a unique hinge.

| Joint | Articulation tensor column | MJWarp DOF column |
|---|---:|---:|
| FR_hip_joint | 9 | 15 |
| FR_thigh_joint | 10 | 16 |
| FR_calf_joint | 11 | 17 |
| FL_hip_joint | 6 | 12 |
| FL_thigh_joint | 7 | 13 |
| FL_calf_joint | 8 | 14 |
| BR_hip_joint | 3 | 9 |
| BR_thigh_joint | 4 | 10 |
| BR_calf_joint | 5 | 11 |
| BL_hip_joint | 0 | 6 |
| BL_thigh_joint | 1 | 7 |
| BL_calf_joint | 2 | 8 |

Calf values are generalized torque about the simulated calf hinge. Converting to physical four-bar motor torque requires the actual linkage transmission/Jacobian, configuration, efficiencies and motor model. This change performs no such conversion and makes no motor-sizing claim.

## Implementation and statistics

New `walking_direct/torque_diagnostics.py` contains the collector and adapter. GPU kernels read solver values and accumulate absolute peaks, sums, near-limit counts and a bounded sample buffer. No source state, controls, reward tensors or autograd graph are written. CPU synchronization occurs only at the iteration boundary, not each physics step.

For each named joint, the denominator is the number of environments times physics samples in that iteration. `MeanAbs_Nm` is the absolute-value mean; `P95Abs_Nm` is the exact 95th percentile with linear interpolation; `EffortLimitFraction` is the fraction with `abs(torque) >= 0.95 * runtime_effort_limit`. `PeakAbs_Nm` is the maximum over the whole interval. `MaxEver_Nm` retains the maximum across intervals and episode resets. Overall peaks take the maximum across all 12 joints.

The existing walking PPO wrapper binds the monitor at algorithm construction, resets run counters, and publishes detached statistics **after** the unchanged optimizer update. `diagnostics.py` inserts them into the last-step log dictionary already retained by native RSL-RL. This occurs once, so the native logger's mean aggregation cannot turn interval maxima into averages of step maxima. It uses the existing native SummaryWriter; no training writer was added. Tests confirm unchanged policy gradients, weights and checkpoint keys.

There are 62 required tags: 12 joints × five groups, plus two overall groups. Two metadata tags record interval and sample count. Twelve optional graphs use `EstimatedTorque/UnclippedComputedPeakAbs_Nm/<joint>`; `TorqueAccumulator(include_computed=False)` omits those graphs. They retain the estimated label even when numerically close to solver output.

The default sample buffer holds 256 physics steps per interval (about 1.2 MB at 100 envs). The unchanged 24-step PPO rollout uses 96. Overflow fails explicitly at the publication boundary instead of silently truncating P95. Increase capacity before graph capture if intentionally increasing rollout length beyond 64 control steps. If future manager-owned decimation prevents observing every physics step, peak graphs are labelled `SampledPeak`/`SampledMaxEver` and the actual sample interval is logged. The current tested configuration samples every physics step.

The adapter deliberately checks the installed backend layout, hinge names, force caps and tensor shapes. It uses a backend-specific solver handle because the public articulation effort API provides estimates. A changed backend/layout fails validation rather than silently relabelling a PD estimate as solver torque. Resetting the environment does not reset run-wide maxima; a new process or explicit run reset does. Torque history remains outside checkpoints.

## Verification artifacts and preservation

- `cpu_tests.log`: **7 tests passed**, including signed data, exact percentile, reset/overflow and identical-input PPO gradient checks.
- `walking_regression.log`: **17 existing focused tests passed**.
- `signal_probe.log`: raw candidate signal values and aliases.
- `runtime_warmed.json`: controlled force reconstruction, reset check and timing samples. Independent rollout equality is explicitly false; no deterministic-physics claim is made.
- `events.json`: per-tag event values for short training runs. All required tags read back exactly once per iteration.
- `train_smoke8.log`, `train_smoke100.log`, `train_viewer8.log`: real Newton training logs and checkpoint output.
- `versions.json`, `changed_files.json`, `changes.diff`, `preservation.json`: versions, exact source changes and protected-file checks.

Before editing, timestamped copies of every changed existing file were saved under `grallator_rl/backups/torque_20261104_125903/`.

Modified existing files: `walking_direct/env.py` (attach/cleanup only), `walking_direct/diagnostics.py` (monitor ownership/publication), `him/walking_ppo.py` (monitor binding and detached publication after optimization). New source/test files: `walking_direct/torque_diagnostics.py`, `scripts/audit_torque_runtime.py`, `tests/test_torque_diagnostics.py`. Audit artifacts and new walking logs are separate from prior runs.

Original standing checkpoint SHA-256 remains:

```
10c9f75518359d9206c66c42f0a51ba6a68f84fcadd4f26c2b323dfd142c8ced
IsaacLab/logs/rsl_rl/grallator_him_standing/2026-11-03_21-06-02_middle_pose_038_resume/model_1700.pt
```

## Launch commands

Setup, using the existing standing-derived warm-start:

```bash
cd /home/poornaprasad/RL/Quadruped/IsaacLab
export PYTHONPATH=/home/poornaprasad/RL/Quadruped:/home/poornaprasad/RL/Quadruped/grallator_rl/src
export GRALLATOR_FREEZE_HIO=0
export GRALLATOR_HIO_LR_OVERRIDE=1e-5
export WALKING_INIT=/home/poornaprasad/RL/Quadruped/grallator_rl/checkpoints/walking_from_standing.pt
```

Eight-env Newton viewer (the audit exercised two iterations; this command requests 25):

```bash
.venv/bin/python ../grallator_rl/scripts/train_walking.py \
  --checkpoint "$WALKING_INIT" --height 0.38 \
  --num-envs 8 --iterations 25 --viz newton_gl --run-name torque_viewer25
```

100-env headless training (the audit exercised three iterations; start the longer run after checking the short test):

```bash
.venv/bin/python ../grallator_rl/scripts/train_walking.py \
  --checkpoint "$WALKING_INIT" --height 0.38 \
  --num-envs 100 --iterations 300 --run-name torque_headless100
```

Existing native TensorBoard writer output:

```bash
.venv/bin/tensorboard \
  --logdir /home/poornaprasad/RL/Quadruped/IsaacLab/logs/rsl_rl/grallator_him_walking_native \
  --host 127.0.0.1 --port 6006
```

Open `http://127.0.0.1:6006` and select the new `torque_*` runs, then `Torque/` and optional `EstimatedTorque/`. Older runs do not acquire new tags retroactively.

Reproduce the telemetry tests and GPU validation:

```bash
.venv/bin/python -m unittest discover -s ../grallator_rl/tests -p test_torque_diagnostics.py -v
.venv/bin/python ../grallator_rl/scripts/run_walking_tests.py
.venv/bin/python ../grallator_rl/scripts/audit_torque_runtime.py \
  --output /home/poornaprasad/RL/Quadruped/grallator_rl/audit/torque_20261104/runtime_recheck.json
```

No long training run, standing retraining, Eureka reward dependency or OpenAI API call was used in this follow-up.
