"""Controlled torque validation and warmed monitoring overhead comparison."""
import argparse
import json
import os
from pathlib import Path
import time

import gymnasium as gym
import torch
import warp as wp
from isaaclab.app import launch_simulation

os.environ.setdefault('GRALLATOR_TARGET_HEIGHT', '0.38')
import grallator_rl.him.walking_registration  # noqa: E402,F401
from grallator_rl.him.walking_env_adapter import WalkingHIMEnvCfg
from grallator_rl.tasks.walking_direct.diagnostics import WalkingDiagnostics
from grallator_rl.tasks.walking_direct.torque_diagnostics import JOINT_NAMES
from isaaclab_newton.physics import NewtonManager


def run(enabled):
    original=WalkingDiagnostics.attach_torque
    if not enabled:
        # Benchmark-only baseline. Production configuration/actuators unchanged.
        WalkingDiagnostics.attach_torque=lambda *args:None
    cfg=WalkingHIMEnvCfg();cfg.scene.num_envs=8;cfg.seed=42
    rewards=[];observations=[]
    try:
        with launch_simulation(cfg,{'headless':True}):
            env=gym.make('GrallatorRl-HIM-Walking-Direct',cfg=cfg).unwrapped
            try:
                env.reset()
                monitor=env.diagnostics.torque
                accumulator=monitor.accumulator if monitor else None
                solver=NewtonManager._solver
                # Warm CUDA graph and compiled kernels; discard startup statistics.
                for _ in range(24):env.step(torch.ones_like(env.actions)*.2)
                if accumulator:
                    accumulator.finish_interval()  # warm quantile/reduction initialization too
                    accumulator.reset_run()
                durations=[];summary_durations=[];largest=0.;computed_largest=0.
                last_metrics=None
                for block in range(5):
                    torch.cuda.synchronize();start=time.perf_counter()
                    for _ in range(24):
                        obs,reward,*_=env.step(torch.ones_like(env.actions)*.2)
                        rewards.append(reward.clone());observations.append(obs['policy'].clone())
                    torch.cuda.synchronize();durations.append(time.perf_counter()-start)
                    if accumulator:
                        assert int(accumulator.count.item())==96
                        start=time.perf_counter();last_metrics=accumulator.finish_interval()
                        torch.cuda.synchronize();summary_durations.append(time.perf_counter()-start)
                        largest=max(largest,float(last_metrics['Torque/OverallMaxEver_Nm']))
                        computed_largest=max(computed_largest,max(float(v) for k,v in last_metrics.items() if k.startswith('EstimatedTorque/')))
                        assert last_metrics['Torque/OverallPeakAbs_Nm']<=100.001
                        for name in JOINT_NAMES:
                            assert 0<=last_metrics['Torque/EffortLimitFraction/'+name]<=1
                        if block==0:
                            ever=accumulator.running.clone()
                            env._reset_idx(torch.tensor([0],device=env.device))
                            torch.testing.assert_close(accumulator.running,ever)
                    elif block==0:
                        env._reset_idx(torch.tensor([0],device=env.device))
                if accumulator:
                    assert largest>0, 'Controlled command produced no solver effort'
                    # Solver output agrees with its native scalar actuator moments;
                    # avoid treating the public PD estimate as validation ground truth.
                    qfrc=wp.to_torch(solver.mjw_data.qfrc_actuator)
                    force=wp.to_torch(solver.mjw_data.actuator_force)
                    model=solver.mj_model
                    for name,col in monitor.mapping.items():
                        joint=next(i for i in range(model.njnt) if int(model.jnt_dofadr[i])==col)
                        actuator_ids=[i for i in range(model.nu) if int(model.actuator_trnid[i,0])==joint]
                        assert len(actuator_ids)==2
                        moments=torch.as_tensor(model.actuator_gear[actuator_ids,0],device=env.device,dtype=torch.float32)
                        expected=(force[:,actuator_ids]*moments).sum(-1).clamp(-100,100)
                        torch.testing.assert_close(qfrc[:,col],expected,atol=2e-4,rtol=1e-5)
                    accumulator.reset_run()
                    assert accumulator.finish_interval()['Torque/OverallMaxEver_Nm']==0
                return dict(enabled=enabled,physics_dt=cfg.sim.dt,control_dt=env.step_dt,
                            collection_seconds=durations,summary_seconds=summary_durations,
                            largest_solver_torque_nm=largest,largest_estimated_unclipped_nm=computed_largest,
                            mapping=monitor.mapping if monitor else None,
                            runtime_mapping=monitor.runtime_mapping if monitor else None,
                            solver_force_reconstruction='PASS' if monitor else None,
                            final_public_applied=env.robot.actuators.applied_effort.torch[0].tolist(),
                            final_solver_qfrc=qfrc[0].tolist() if monitor else None),torch.stack(rewards).cpu(),torch.stack(observations).cpu()
            finally:env.close()
    finally:WalkingDiagnostics.attach_torque=original


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Fresh envs use the same seed and scripted actions; monitoring consumes no RNG.
    baseline,base_rewards,base_obs=run(False)
    monitored,torque_rewards,torque_obs=run(True)
    # Independent contact-solver rollouts need not be bitwise deterministic.
    # Report this comparison; never turn monitoring validation into a claim of
    # deterministic physics. Source/optimizer regression tests cover interfaces.
    equal=torch.equal(torque_rewards,base_rewards) and torch.equal(torque_obs,base_obs)
    baseline_ms=1000*sum(baseline['collection_seconds'])/5
    monitored_ms=1000*sum(monitored['collection_seconds'])/5
    summary_ms=1000*sum(monitored['summary_seconds'])/5
    result=dict(baseline=baseline,monitored=monitored,
                mean_24_control_steps_ms={'baseline':baseline_ms,'monitored':monitored_ms,'summary':summary_ms},
                collection_overhead_pct=100*(monitored_ms/baseline_ms-1),
                independent_rollouts_bitwise_equal=equal,
                independent_rollout_max_reward_difference=(torque_rewards-base_rewards).abs().max().item(),
                independent_rollout_max_observation_difference=(torque_obs-base_obs).abs().max().item())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print('TORQUE_RUNTIME_PASS',json.dumps(result),flush=True)


if __name__=='__main__':main()
