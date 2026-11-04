"""Real Newton contact/observation/reset preflight. No optimization; airtime off.

Run before training after a sensor/backend/geometry change. Writes measured
evidence, checks forced synchronous 599-step timeout, and isolates a lifted robot
to distinguish contact force from a pose-based proxy.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import gymnasium as gym
import torch
from tensordict import TensorDict
from isaaclab.app import launch_simulation

os.environ.setdefault('GRALLATOR_TARGET_HEIGHT', '0.38')
import grallator_rl.him.walking_registration  # noqa: E402,F401
from grallator_rl.him.walking_env_adapter import WalkingHIMEnvCfg
from grallator_rl.him.rsl_actor import HIMRslActorModel
from grallator_rl.him.walking_ppo_cfg import WalkingPPORunnerCfg
from grallator_rl.him.walking_checkpoint import WARM_START, verify_warm_start
from grallator_rl.tasks.standing_direct.config.grallator.env import StandingEnv
from grallator.config.observations import build_policy_observation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs',type=int,default=16)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    provenance=verify_warm_start()
    cfg=WalkingHIMEnvCfg(); cfg.scene.num_envs=args.num_envs
    cfg.rewards.feet_air_time=0.0  # validate signal before enabling its reward
    with launch_simulation(cfg, {'headless':True}):
        env=gym.make('GrallatorRl-HIM-Walking-Direct',cfg=cfg).unwrapped
        try:
            with torch.inference_mode():
                obs,_=env.reset()
                assert obs['policy'].shape==(args.num_envs,271) and obs['critic'].shape==(args.num_envs,92)
                rc=WalkingPPORunnerCfg()
                actor=HIMRslActorModel(TensorDict(obs,[args.num_envs]),rc.obs_groups,'actor',12,
                                      distribution_cfg=rc.actor.distribution_cfg.to_dict()).to(env.device)
                actor.load_state_dict(torch.load(WARM_START,map_location=env.device,weights_only=True)['actor_state_dict'],strict=True)
                env.velocity_commands[:]=torch.tensor([.2,.03,.04],device=env.device)
                obs=env._get_observations()
                current=obs['policy'][:,:45]
                torch.testing.assert_close(current[:,6:9],env.velocity_commands)
                torch.testing.assert_close(StandingEnv._get_observations(env)['policy'], current)
                expected=build_policy_observation(env.robot.data.root_ang_vel_b.torch,
                    env.robot.data.projected_gravity_b.torch,env.velocity_commands,
                    env.robot.data.joint_pos.torch-env.middle_joint_pos,env.robot.data.joint_vel.torch,env.actions)
                torch.testing.assert_close(current,expected)
                torch.testing.assert_close(obs['critic'][:,:45],current)
                torch.testing.assert_close(obs['critic'][:,45:48],env.robot.data.root_lin_vel_b.torch)
                torch.testing.assert_close(obs['policy'][:,270],(env.height_command[:,0]-cfg.middle_base_height)/.1)
                # Mapping is by actual tensor joint names, checked against original reference.
                for name,angle in cfg.middle_joint_pos.items():
                    idx=env.robot.joint_names.index(name)
                    torch.testing.assert_close(env.middle_joint_pos[:,idx],torch.full_like(env.middle_joint_pos[:,idx],angle))
                env._pre_physics_step(torch.ones_like(env.actions)*.1)
                torch.testing.assert_close(env.joint_targets,env.middle_joint_pos+.1*env.action_scales)
                env.reset()
                max_force=0.; contact_samples=0; resets=0
                for step in range(180):
                    command_used=env.velocity_commands.clone()
                    obs,reward,terminated,truncated,extras=env.step(actor(TensorDict(obs,[args.num_envs])))
                    assert torch.isfinite(reward).all() and all(torch.isfinite(v).all() for v in obs.values())
                    forces=env.contact_sensor.data.net_normal_forces_w.torch[:,env._foot_sensor_ids,:].norm(dim=-1)
                    max_force=max(max_force,forces.max().item())
                    contact_samples+=int((forces>cfg.rewards.contact_force_threshold_n).sum())
                    resets+=int((terminated|truncated).sum())
                    torch.testing.assert_close(obs['policy'][:,6:9],env.velocity_commands)
                    expected_track=cfg.rewards.track_xy*torch.exp(
                        -(env.robot.data.root_lin_vel_b.torch[:,:2]-command_used[:,:2]).square().sum(-1)
                        / cfg.rewards.xy_sigma**2)
                    torch.testing.assert_close(extras['log']['Reward_Rate/track_xy'],expected_track.mean())
                assert contact_samples>0, 'No measured foot support in 180 control steps'
                # Known free-flight state: lift away from flat ground, zero velocity.
                pose=env.robot.data.root_pose_w.torch.clone(); pose[:,2]+=1.
                ids=torch.arange(args.num_envs,device=env.device,dtype=torch.int32)
                env.robot.write_root_pose_to_sim_index(root_pose=pose,env_ids=ids)
                env.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros_like(env.robot.data.root_vel_w.torch),env_ids=ids)
                for _ in range(3): env.step(torch.zeros_like(env.actions))
                lifted_force=env.contact_sensor.data.net_normal_forces_w.torch[:,env._foot_sensor_ids,:].norm(dim=-1)
                assert not (lifted_force>cfg.rewards.contact_force_threshold_n).any(), 'Lifted feet report contact'
                env.reset()
                # Synchronous 599/600 timeout regression: DirectRLEnv increments
                # before dones and standing uses >= max_episode_length-1.
                env.episode_length_buf[:]=env.max_episode_length-2
                obs,reward,terminated,truncated,extras=env.step(torch.zeros_like(env.actions))
                assert truncated.all() and (env.episode_length_buf==0).all()
                assert not env.gait.valid_swing.any() and not env.gait.seen_contact.any()
                h=obs['policy'][:,:270].reshape(args.num_envs,6,45)
                torch.testing.assert_close(h,h[:,:1].expand_as(h))
                assert 'Episode_Reward/feet_air_time' in extras['log']
                assert extras['log']['Episode_Reward/feet_air_time']==0
                assert 'Reward_Rate/feet_air_time' in extras['log']
                total=sum(v for k,v in extras['log'].items()
                          if k.startswith('Episode_Reward/') and k!='Episode_Reward/total')
                torch.testing.assert_close(total,extras['log']['Episode_Reward/total'])
                # Exercise the installed native logger with actual task extras
                # from a controlled completed episode, then read its event file.
                from rsl_rl.utils.logger import Logger
                from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
                event_dir=args.output.parent / ('controlled_events_' + str(args.num_envs))
                logger=Logger(str(event_dir), {'algorithm':{},'num_steps_per_env':1,'logger':'tensorboard'},
                              {},args.num_envs,False,1,0,env.device)
                logger.init_logging_writer()
                logger.process_env_step(reward,terminated|truncated,extras)
                logger.log(0,0,1,.02,.001,{},3e-4,torch.full((12,),.12),None,print_minimal=True)
                logger.stop_logging_writer()
                events=EventAccumulator(str(event_dir)).Reload()
                tags=events.Tags()['scalars']
                for tag in ('Reward_Rate/feet_air_time','Episode_Reward/feet_air_time',
                            'Reward_Rate/total','Episode_Reward/total'):
                    assert tag in tags,tag
                # Repeated observation queries cannot resample a boundary twice.
                env.episode_length_buf[:]=env._command_resample_steps
                env._get_observations(); cmd=env.velocity_commands.clone();env._get_observations()
                torch.testing.assert_close(env.velocity_commands,cmd)
                from isaaclab.utils.math import quat_from_euler_xyz
                root=env.robot.data.root_pose_w.torch.clone()
                zero=torch.zeros(args.num_envs,device=env.device)
                root[:,3:]=quat_from_euler_xyz(zero,zero,torch.full_like(zero,torch.pi/2))
                velocity=torch.zeros_like(env.robot.data.root_vel_w.torch);velocity[:,1]=.2
                env.robot.write_root_pose_to_sim_index(root_pose=root,env_ids=ids)
                env.robot.write_root_velocity_to_sim_index(root_velocity=velocity,env_ids=ids)
                torch.testing.assert_close(env.robot.data.root_lin_vel_b.torch,
                    torch.tensor([.2,0.,0.],device=env.device).expand(args.num_envs,-1),atol=1e-6,rtol=1e-5)
                # Python loads installed editable-package finders at startup;
                # their presence is not an import/dependency on that package.
                active_eureka=[name for name in sys.modules if name.startswith(
                    ('isaaclab_eureka','eureka_grallator','grallator_rl.eureka_reward'))]
                assert not active_eureka,active_eureka
                evidence=dict(provenance=provenance,num_envs=args.num_envs,step_dt=env.step_dt,
                    backend=type(cfg.sim.physics).__name__,solver=str(cfg.sim.physics.solver_cfg.class_type),
                    physics_handles_decimation=env._physics_handles_decimation,
                    sensor_update_period=cfg.scene.contact_forces.update_period,
                    sensor_history_length=cfg.scene.contact_forces.history_length,
                    sensor_names=list(env.contact_sensor.sensor_names),foot_ids=env._foot_sensor_ids,
                    max_support_force_n=max_force,contact_samples=contact_samples,
                    lifted_max_force_n=lifted_force.max().item(),resets_during_180_steps=resets,
                    timeout_step=env.max_episode_length-1,timeout_all=True,
                    observations_values='PASS',action_mapping='PASS',history_reset='PASS',
                    contact_signal='PASS',body_frame_velocity='PASS',eureka_modules_loaded=False,
                    reward_airtime_during_preflight=0.,
                    tensorboard_tags=tags,event_dir=str(event_dir.resolve()))
                args.output.parent.mkdir(parents=True,exist_ok=True)
                args.output.write_text(json.dumps(evidence,indent=2)+'\n')
                print('RUNTIME_PREFLIGHT_PASS',json.dumps(evidence),flush=True)
        finally:
            env.close()


if __name__=='__main__': main()
