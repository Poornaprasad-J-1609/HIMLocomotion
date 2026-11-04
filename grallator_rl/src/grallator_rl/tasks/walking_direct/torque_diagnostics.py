"""Read-only MJWarp joint-side torque monitoring, independent of rewards.

The public implicit-actuator effort telemetry is an estimate. This adapter reads
MJWarp qfrc_actuator *after* solving instead; it is solver-reported generalized
actuation torque, not hardware measurement or four-bar motor shaft torque.
"""
from __future__ import annotations

import torch
import warp as wp

JOINT_NAMES = tuple(f'{leg}_{part}_joint' for leg in ('FR', 'FL', 'BR', 'BL')
                    for part in ('hip', 'thigh', 'calf'))


def joint_indices(names):
    """Reject missing, duplicated or extra joints; never assume tensor order."""
    if len(names) != 12 or len(set(names)) != 12 or set(names) != set(JOINT_NAMES):
        raise ValueError(f'Expected exactly the 12 Grallator joints, got {names}')
    return [names.index(name) for name in JOINT_NAMES]


@wp.kernel(enable_backward=False)
def _sample(source: wp.array2d(dtype=wp.float32), columns: wp.array(dtype=wp.int32),
            computed: wp.array2d(dtype=wp.float32), computed_columns: wp.array(dtype=wp.int32),
            limits: wp.array2d(dtype=wp.float32), count: wp.array(dtype=wp.int32),
            samples: wp.array2d(dtype=wp.float32), peaks: wp.array2d(dtype=wp.float32),
            sums: wp.array2d(dtype=wp.float32), near: wp.array2d(dtype=wp.float32),
            computed_peaks: wp.array2d(dtype=wp.float32), invalid: wp.array(dtype=wp.int32)):
    env, joint = wp.tid()
    value = wp.abs(source[env, columns[joint]])
    estimate = wp.abs(computed[env, computed_columns[joint]])
    if not wp.isfinite(value) or not wp.isfinite(estimate):
        wp.atomic_max(invalid, 0, 1)
    if count[0] < samples.shape[0]:
        samples[count[0], env * 12 + joint] = value
    else:
        wp.atomic_max(invalid, 0, 2)
    peaks[env, joint] = wp.max(peaks[env, joint], value)
    computed_peaks[env, joint] = wp.max(computed_peaks[env, joint], estimate)
    sums[env, joint] += value
    if value >= 0.95 * limits[env, joint]:
        near[env, joint] += 1.0


@wp.kernel(enable_backward=False)
def _advance(count: wp.array(dtype=wp.int32)):
    count[0] += 1


class TorqueAccumulator:
    """GPU interval statistics; episode resets deliberately do not clear them.

    All sample buffers have fixed addresses for CUDA graph replay. P95 is exact
    (linear interpolation) over all env/time samples, computed once per rollout.
    Capacity overflow and nonfinite samples fail at the logging boundary.
    """
    def __init__(self, num_envs, limits, *, capacity=256, sample_dt=.005, sampled_peak=False,
                 include_computed=True):
        if limits.shape != (num_envs, 12) or limits.dtype != torch.float32:
            raise ValueError('Expected float32 effort limits [num_envs,12] in reporting order')
        if not torch.isfinite(limits).all() or not (limits > 0).all() or capacity < 1 or sample_dt <= 0:
            raise ValueError('Invalid effort limits, sample capacity or sample interval')
        self.num_envs, self.capacity, self.sample_dt = num_envs, capacity, sample_dt
        self.sampled_peak = sampled_peak
        self.include_computed = include_computed
        self.limits = limits.detach().clone().contiguous()
        self.samples = limits.new_zeros((capacity, num_envs * 12))
        self.peaks = torch.zeros_like(limits)
        self.sums = torch.zeros_like(limits)
        self.near = torch.zeros_like(limits)
        self.computed_peaks = torch.zeros_like(limits)
        self.running = limits.new_zeros(12)
        self.count = torch.zeros(1, dtype=torch.int32, device=limits.device)
        self.invalid = torch.zeros_like(self.count)

    def sample(self, source, columns, computed, computed_columns):
        """Graph-safe launch; no host readbacks or Torch allocations per sample."""
        wp.launch(_sample, dim=(self.num_envs, 12), inputs=[
            source, columns, computed, computed_columns, wp.from_torch(self.limits),
            wp.from_torch(self.count), wp.from_torch(self.samples), wp.from_torch(self.peaks),
            wp.from_torch(self.sums), wp.from_torch(self.near), wp.from_torch(self.computed_peaks),
            wp.from_torch(self.invalid)], device=str(self.limits.device))
        wp.launch(_advance, dim=1, inputs=[wp.from_torch(self.count)], device=str(self.limits.device))

    def reset_interval(self):
        for value in (self.count, self.invalid, self.peaks, self.sums, self.near, self.computed_peaks):
            value.zero_()

    def reset_run(self):
        self.reset_interval()
        self.running.zero_()

    @torch.no_grad()
    def finish_interval(self):
        # The only monitoring readback is here, once per PPO iteration.
        count, invalid = int(self.count.item()), int(self.invalid.item())
        if invalid or count > self.capacity:
            raise RuntimeError(f'Torque telemetry invalid/overflow: code={invalid}, steps={count}, capacity={self.capacity}')
        peak = self.peaks.amax(0)
        self.running.copy_(torch.maximum(self.running, peak))
        running = self.running.clone()
        mean = self.sums.sum(0) / max(1, count * self.num_envs)
        fraction = self.near.sum(0) / max(1, count * self.num_envs)
        p95 = (torch.quantile(self.samples[:count].reshape(-1, 12), .95, dim=0)
               if count else torch.zeros_like(peak))
        computed_peak = self.computed_peaks.amax(0)
        peak_name = 'SampledPeakAbs_Nm' if self.sampled_peak else 'PeakAbs_Nm'
        ever_name = 'SampledMaxEver_Nm' if self.sampled_peak else 'MaxEver_Nm'
        metrics = {}
        for j, name in enumerate(JOINT_NAMES):
            for group, value in ((peak_name, peak), (ever_name, running), ('MeanAbs_Nm', mean),
                                 ('P95Abs_Nm', p95), ('EffortLimitFraction', fraction)):
                metrics[f'Torque/{group}/{name}'] = value[j]
            if self.include_computed:
                computed_group = ('UnclippedComputedSampledPeakAbs_Nm' if self.sampled_peak
                                  else 'UnclippedComputedPeakAbs_Nm')
                metrics[f'EstimatedTorque/{computed_group}/{name}'] = computed_peak[j]
        metrics[f'Torque/Overall{peak_name}'] = peak.amax()
        metrics[f'Torque/Overall{ever_name}'] = running.amax()
        metrics['Torque/SampleInterval_s'] = peak.new_tensor(self.sample_dt)
        metrics['Torque/PhysicsSamplesPerJoint'] = peak.new_tensor(count * self.num_envs)
        self.reset_interval()
        return metrics


class NewtonTorqueDiagnostics:
    """Layout-checked, read-only adapter to the installed MJWarp solver.

    MJWarp's 24 scalar P/D actuator entries are not the 12 reported joints.
    qfrc_actuator already combines those entries into 12 hinge DOF torques.
    """
    def __init__(self, robot, sim):
        from isaaclab_newton.physics import NewtonManager
        self.manager = NewtonManager
        solver = self.manager._solver  # Backend-specific API, verified against installed Newton 1.6.1.
        if solver is None or solver.use_mujoco_cpu:
            raise RuntimeError('Torque monitoring requires the verified Newton MJWarp GPU backend')
        runtime_ids = joint_indices(list(robot.joint_names))
        model = solver.mj_model
        mj_ids = []
        for name in JOINT_NAMES:
            matches = [i for i in range(model.njnt) if model.joint(i).name.endswith('_' + name)]
            if len(matches) != 1 or int(model.jnt_type[matches[0]]) != 3:  # MuJoCo hinge
                raise ValueError(f'Expected exactly one MJWarp hinge for {name}: {matches}')
            mj_ids.append(int(model.jnt_dofadr[matches[0]]))
        self.source = solver.mjw_data.qfrc_actuator
        self.computed = robot.actuators.computed_effort.warp
        if self.source.shape != (robot.num_instances, model.nv) or len(set(mj_ids)) != 12:
            raise ValueError('Unexpected MJWarp world/DOF layout')
        if self.computed.shape != (robot.num_instances, 12) or not robot.actuators.has_implicit_actuators:
            raise ValueError('Unexpected Grallator implicit-actuator layout')
        # Validate the actual solver's joint-level force limits, not just config values.
        limits = robot.data.joint_effort_limits.torch[:, runtime_ids].contiguous()
        for j, name in enumerate(JOINT_NAMES):
            idx = next(i for i in range(model.njnt) if int(model.jnt_dofadr[i]) == mj_ids[j])
            if not model.jnt_actfrclimited[idx]:
                raise ValueError(f'Solver force limiting is disabled for {name}')
            actual = torch.as_tensor(model.jnt_actfrcrange[idx], device=limits.device)
            if not torch.allclose(actual, torch.stack((-limits[0, j], limits[0, j])).to(actual.dtype)):
                raise ValueError(f'Solver effort range differs for {name}')
        self.columns = wp.array(mj_ids, dtype=wp.int32, device=str(limits.device))
        self.computed_columns = wp.array(runtime_ids, dtype=wp.int32, device=str(limits.device))
        self.accumulator = TorqueAccumulator(
            robot.num_instances, limits, sample_dt=float(sim.cfg.dt) * self.manager._decimation,
            sampled_peak=self.manager._decimation != 1)
        self.mapping = dict(zip(JOINT_NAMES, mj_ids))
        self.runtime_mapping = dict(zip(JOINT_NAMES, runtime_ids))
        self._callback = self._sample
        # Fixed buffers are registered before the first training-step capture.
        if self.manager._graph is not None:
            raise RuntimeError('Torque callback must be installed before CUDA graph capture')
        self.manager.register_post_step_callback(self._callback)

    def _sample(self):
        self.accumulator.sample(self.source, self.columns, self.computed, self.computed_columns)

    def close(self):
        self.manager.unregister_post_step_callback(self._callback)
