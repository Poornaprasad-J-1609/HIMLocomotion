# Fixes in the single-height standing refactor

**Source:** `grallator_rl.zip` uploaded on 2026-11-02. The original upload is unchanged; this is a new cleaned package.

1. Rebuilt a **truncated `env.py`** (the upload stopped inside `StandingEnv.__init__()` at the symmetry-pairs list, with no actions, observations, rewards, resets or termination methods).
2. Moved reward weights to `config/grallator/reward_cfg.py` and pure reward calculations to `standing_direct/rewards.py` so changes to rewards don't alter the HIM observation layout.
3. Replaced ambiguous `crouch_base_height` / `crouch_joint_pos` with **exact `middle_base_height=0.428` / 12D `middle_joint_pos`**. Initial joint positions and velocities are set explicitly; no random initial pose/tilt.
4. Removed **both** `height_command_min` and `height_command_max` and any target randomization. All environments track one `target_base_height`, preserved from the ZIP at 0.50 m by default. `scripts/train_standing.py --height` makes it easy to supply a different one.
5. Preserved the working **45D base observation / 270D HIM encoder / 271D PPO actor observation / 92D critic observation**. Height is a separate normalized scalar `(height_command - middle_base_height)/0.10`, with no clipping or min/max normalization.
6. PPO action mapping uses `MIDDLE_POSE + (0.06 hip / 0.25 thigh and calf) * action`, clipped to actual joint limits.
7. Removed a reward-design conflict: the leg pose deviation penalty is weaker and excludes hip joints; the separate hip stability penalty remains. All named reward terms are accumulated **once** per control step, and dense height progress is relative to the sole requested target.
8. Kept non-foot contact force penalty and logging; uses Newton's contact sensor `sensor_names` and `net_normal_forces_w`, subject to the real USD link collision geometry and aggregate-contact caveat.
9. Fixed `HIMTrainer.update_velocity()` requiring 271D input (incorrect) when the HIM encoder accepts 270D. Ensured PPO optimizer excludes parameters belonging to the HIO estimator optimizer.
10. Preserved checkpoint files and the main training entry points, added runner and CPU math tests. Removed Python caches, `.git` internals, and `.bak` files **from the new distribution only**, not from the original uploaded project.

**Not validated here:** end-to-end Isaac Lab Newton simulation, robot asset collision geometry, 0.50 m reachability, contact-force values, policy convergence, or hardware safety. Sim-to-real deployment needs actuator, four-bar, and joint limit validation.
