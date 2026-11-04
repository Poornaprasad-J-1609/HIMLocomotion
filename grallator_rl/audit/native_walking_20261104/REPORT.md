# Native Grallator walking audit — 4 November 2026

Walking now has native, stateful touchdown airtime, structured native TensorBoard logging, and a mandatory verified standing warm-start. The nine existing walking reward terms, their coefficients, robot configuration, standing code and shared HIM/PPO source were preserved. This audit verifies execution and connectivity; it does not establish a successful walking gait.

## Test matrix

Paths below are relative to `/home/poornaprasad/RL/Quadruped`.

| Check | Result | Executed evidence / correction |
|---|---|---|
| Actual import and Gym registration | PASS | Real `IsaacLab/.venv/bin/python` resolves `tasks/walking_direct/env.py` in this checkout. `him/walking_registration.py:7` registers `WalkingHIMEnv`; new walking runner config selects the native on-policy runner and walking-only PPO wrapper. Runtime JSON records NewtonMJWarpManager. |
| Newton stepping, 16/64 headless | PASS | Three PPO iterations with 16 envs, two with 64; finite observations, rewards, event scalars and losses; saved checkpoints. See `train_final16.log`, `train_final64.log`, and training evidence JSON files. |
| Eight-env graphical Newton | PASS | `--viz newton_gl`, two training iterations, successful process exit and checkpoint. See `train_finalviewer8.log`. Viewer launch/render execution tested; visual gait quality was not assessed. |
| Observation values, ordering and commands | PASS | Real runtime compares standing and walking 45D proprioception elementwise, command slots 6:9, critic velocity slots 45:48, and appended normalized height. Actor 271, critic 92. A 90-degree root yaw transforms world `(0,.2,0)` to body `(.2,0,0)`. `runtime_final16.json`; `scripts/audit_walking_runtime.py`. |
| History, action mapping and resets | PASS | CPU tests check newest-first FIFO, fill-on-first-observation and selective reset. Runtime verifies actual tensor joint names, MIDDLE_POSE and action offsets, plus all-env timeout at step 599 followed by clean history/gait reset. Command-resampling boundary tested against actual tracking reward values, including repeated observation queries. |
| Source checkpoint and warm-start | PASS | SHA-256 below; strict actor loading; every actor/critic/estimator tensor and normalization buffer compared. Only actor exploration std differs. Empty PPO/HIO optimizer states and iteration 0 verified. No checkpoint rebuild or overwrite. |
| PPO actor/critic updates | PASS | Real RSL-RL PPO CPU rollouts exercise actor and critic gradients, disjoint optimizer ownership, clipped surrogate/value loss, entropy and KL. Numeric GAE and timeout bootstrap tests pass. GPU checkpoint comparisons confirm changed actor/critic tensors. |
| HIO active/frozen | PASS | Source encoder, target encoder and prototypes have nonzero gradients during HIO. Velocity and swap losses execute with LR `1e-5`. CPU freeze leaves estimator unchanged. GPU resumed frozen run also leaves estimator bitwise unchanged while actor/critic change. |
| PPO/HIO KL stability | PASS for short connectivity tests; NOT VERIFIED for long training | Initial inherited PPO LR `3e-4` produced first-update KL about 8. Walking alone now starts at `1e-5`. Final 16-env PPO KL `0.041974, 0.021166, 0.038565`; post-HIO `0.041870, 0.021041, 0.038887`. Native adaptive schedule remains enabled, so LR can subsequently increase. |
| Contact signal and airtime | PASS | Actual foot sensor indices `[16,12,8,4]` correspond to FR, FL, BR, BL. Support forces reached about 344.54 N; deliberately lifted robot had 0 N foot force. Preflight disabled the airtime weight while validating the signal. CPU tests verify qualifying swing, threshold, duration bounds, flight rejection, no standing/reset payment and no duplicate touchdown payment. |
| TensorBoard routing | PASS | Installed RSL-RL `Logger.process_env_step -> Logger.log -> SummaryWriter`; EventAccumulator reads actual events. Training has `Reward_Rate/feet_air_time=0`. Controlled real-env timeout test has `Episode_Reward/feet_air_time=0` and matching episode total. Short training runs need not contain completed episodes. |
| Syntax + focused CPU suite | PASS | `cpu_tests.log`: 17 tests, OK; AST syntax checks cover all project Python source/scripts/tests. No pytest was installed; runnable unittest entrypoint supplied. |
| Entire existing test suite | FAIL — pre-existing standing fixture mismatch | `all_tests.log`: 25 tests, 22 pass, three standing tests error because their fixture omits the existing required `front_back_pairs` argument. Standing files and these original tests were left unchanged. |
| Torque/power, clearance and impact physical signals | NOT VERIFIED / blocked | Shadow-PD telemetry is not validated solver effort; nonzero torque/power weights are rejected. Clearance needs a verified sole/terrain clearance reference; impact needs validated impulse/peak-force sampling. No fabricated measurements or enabled surrogate terms. |
| Learned walking quality / 300+ iterations | NOT VERIFIED | No long training run was launched. Smoke tests establish connectivity, not task success. |

## Checkpoint preservation and actual architecture

Trusted original:

```
IsaacLab/logs/rsl_rl/grallator_him_standing/2026-11-03_21-06-02_middle_pose_038_resume/model_1700.pt
SHA-256: 10c9f75518359d9206c66c42f0a51ba6a68f84fcadd4f26c2b323dfd142c8ced
```

Selected existing warm-start: `grallator_rl/checkpoints/walking_from_standing.pt`. `walking_init_std012.pt` and the old Eureka-directory candidate also matched the source except exploration std, but walking uses the native checkpoint path. Original std mean is `0.0195510` (range approximately `0.01029–0.03702`), warm-start std is `0.12` in all 12 actions. All network weights and critic normalization buffers match exactly; both optimizer states are empty; iteration is zero. `him/walking_checkpoint.py:15` performs this read-only verification on every fresh walking launch. Fresh launch skips loading optimizer state; `--resume` explicitly restores walking optimizer/iteration state. Native RSL-RL strict tensor loading remains enabled.

The first sandboxed smoke attempt reported no CUDA device (`baseline_smoke.log`). Checking outside the sandbox found the RTX 4060 and all subsequent GPU tests completed; no driver, Isaac Sim or backend replacement was required. Walking's launcher now defaults to the source checkpoint's 0.38 m target instead of its prior 0.40 m default; height remains explicitly configurable.

Actor: six newest-first 45D observations plus one height scalar, 271 total. The estimator sees 270D history and produces 3D velocity plus 16D normalized latent. Actor MLP input is current 45D proprioception + velocity + latent + height = 65; hidden layers `[512,256,128]`; 12 outputs. Critic input is the original 91 privileged values + height = 92, with its original empirical normalization and `[512,256,128]` hidden layers. Proprioception uses the shared raw ordering: angular velocity 0:3, projected gravity 3:6, body-frame commands 6:9, joint offsets 9:21, joint velocity 21:33, previous applied action 33:45. Height is `(command - middle_base_height)/0.10`.

The actual standing config currently resets at **0.270 m**, although older launcher comments mention 0.428 m. Both hip and leg action scales are **0.25 rad**. PD parameters remain Kp=250, Kd=4, effort limit=100. These existing values were preserved, not reconciled by silently changing robot behavior. Geometry, joint ordering, four-bar-related assets and the original standing launcher were not edited. The original checkpoint remains separately usable by the original standing task.

## Changes and reward units

The old grep was empty because the current native files had contact diagnostics but no `feet_air_time` coefficient, completed-swing state or reward integration. Importing the real registered task proved that this was the executed source. It was a partial implementation, not an absent installation or wrong module. Old JSON/candidate experiments remain on disk; no native walking reward reads them, their environment variable, an API, or Eureka modules. An installed Python editable-package finder with “eureka” in its name is startup packaging metadata, not an imported reward dependency; the runtime asserts actual Eureka package modules are absent.

Baseline reward-rate coefficients remain: `track_xy=4`, `track_yaw=1`, `height=.5`, `upright=.6`, `vertical_velocity=.4`, `joint_velocity=.01`, `action_rate=.05`, `action_magnitude=.01`, `undesired_contact=2`. Their arithmetic/sign/normalization is unchanged in `walking_direct/rewards.py:14`.

`gait_state.py:36` tracks observed support -> liftoff -> touchdown independently per foot/env. A swing must remain moving, upright, free of undesired contacts and supported by at least two feet. Valid duration is 0.08–0.35 s; first six steps and terminal steps are excluded. Loss of support invalidates the whole swing. Reset clears contact arming, timers and history, including synchronous step-599 timeouts. The moving threshold is planar command norm >0.10 m/s; yaw-only commands do not qualify.

Each valid touchdown earns **weight × completed swing seconds**, once. With weight 2, a 0.10 s swing earns 0.20 reward units. `rewards.py:61` converts the event amount to a rate by dividing by control `step_dt`; `env.py:118` multiplies the rate sum by `step_dt` once. A single-foot event is bounded by 0.70 reward units. No payment occurs for continued airborne time, repeated contact, standing commands, invalid flight, overlong swings or reset touchdown.

The current normal-force sensor buffer is updated at each 0.005 s physics step (`update_period=0`, history length 1; runtime confirms physics does not own decimation). Gait state samples it once per 0.020 s control interval. It does **not** reconstruct substep-only transitions. The 2 N threshold uses the norm of measured world normal force, not friction or guessed foot height. Aggregate sensors can include self/obstacle contact; the undesired-contact stability gate provides an additional safeguard, but this is not a ground-filtered force sensor.

Optional zero-weight terms are contact-gated stance planar slip, binary diagonal contact synchronization (FL+BR versus FR+BL), variance of completed air/stance times, roll/pitch angular velocity, reset-masked world vertical acceleration, joint acceleration and MIDDLE_POSE deviation (5× cost for standing commands). The trot term is a switchable binary phase bias, not a proven optimal gait or a full continuous phase controller. Torque/power pure math exists for CPU checking but runtime activation is blocked pending trustworthy effort telemetry. Clearance and impact are deliberately absent for the reasons in the table.

The optional terms draw on the categories in the public [Isaac Lab Spot research configuration](https://github.com/isaac-sim/IsaacLab/blob/release/3.0.0-beta2/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/flat_env_cfg.py) and [public Spot reward code](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/contrib/velocity/config/spot/mdp/rewards.py). They are original native Torch implementations with no copied coefficient set and no claim about Boston Dynamics production control.

`diagnostics.py:20` constructs fresh scalar snapshots so RSL-RL can retain each step's dictionary safely. Reward rates, full episode sums and totals are separate. Instantaneous raw tracking and reset-excluded quality samples are separate; eligible episode aggregates reset per environment. Old `Diagnostics/*` tag names are retained; legacy moving/cumulative tags now exclude reset transients and reset per environment. Episode tags appear only for actual resets. No competing writer was added to training; the controlled event test explicitly drives the installed native logger.

Command resampling now happens before the next observation, after the preceding action's reward. Previously it happened at pre-physics time and could change a command after the actor had already chosen its action. Repeated observation reads at a boundary cannot resample twice.

Shared `him_ppo.py`, `rsl_actor.py`, estimator, trainer and standing runner config are unchanged. Walking uses a minimal `WalkingPPO` subclass (`him/walking_ppo.py:6`) to clear residual HIO gradients before native PPO clips all actor parameters. Otherwise the previous HIO gradients could enter the next actor clipping norm even though estimator parameters are excluded from PPO's optimizer. A CPU regression test injects stale gradients and verifies they are cleared; PPO math and separate HIO update remain inherited.

## Actual training log excerpt

From `train_final16.log` (3 iterations, initialized from the verified native warm-start):

```
[HIM PPO] LR=1.000000e-05 | PPO KL=0.041974 | Post-HIO KL=0.041870
[HIM PPO] LR=1.000000e-05 | PPO KL=0.021166 | Post-HIO KL=0.021041
[HIM PPO] LR=1.000000e-05 | PPO KL=0.038565 | Post-HIO KL=0.038887
Training time: 2.15 seconds
```

This avoids the observed initial KL≈8 spike, but KL remains above the desired 0.01 on some short updates. HIO-induced KL changes were small in these tests. Monitor both `Loss/ppo_kl` and `Loss/post_hio_kl`, plus tracking/falls/undesired contacts, during the requested 25-iteration test before longer training. No automatic long-run quality or convergence claim is made. Airtime remained zero during the short physical training runs; nonzero completed-swing arithmetic was exercised in controlled CPU tests.

## Files, backups and reproduction

Backup: `grallator_rl/backups/native_audit_20261104_120926/`. Every pre-existing walking file edited was copied there before editing. Initial root git status showed project trees untracked; no resets, deletions, commits or overwrites of prior experiments were performed.

Modified existing files:

- `src/grallator_rl/tasks/walking_direct/{env.py,env_cfg.py,reward_cfg.py,rewards.py}`
- `src/grallator_rl/him/walking_registration.py`
- `scripts/train_walking.py`

New files under `grallator_rl/`:

- `src/grallator_rl/tasks/walking_direct/{gait_state.py,diagnostics.py}`
- `src/grallator_rl/him/{walking_checkpoint.py,walking_ppo_cfg.py,walking_ppo.py}`
- `scripts/{audit_walking_runtime.py,inspect_walking_run.py,run_walking_tests.py}`
- `tests/test_native_walking.py`
- This audit directory, manifests, logs and controlled native event files; separate native walking training runs under `IsaacLab/logs/rsl_rl/grallator_him_walking_native/`.

`changes.diff` contains the exact source diff. `changed_files.json` records changed-file hashes. `preservation_sha256.json` and `preservation_after.json` verify 26 original checkpoint/source files unchanged; the only expected changed entry in that pre-existing HIM list is walking registration. `training_summary.json` and per-run evidence contain event scalars, optimizer state counts and changed network tensor names.

Copy-paste commands for setup, 25-iteration graphical smoke, conditional 300-iteration headless training, TensorBoard, editing weights, CPU/GPU tests and inspecting the report are in [COMMANDS.md](COMMANDS.md). A long training run is intentionally left for after the graphical smoke and its diagnostics pass.
