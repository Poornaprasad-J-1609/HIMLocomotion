"""Run with unittest; CPU tests need no Isaac Sim, pytest, or CUDA."""
import copy
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict
from grallator_rl.tasks.walking_direct.reward_cfg import WalkingRewardCfg
from grallator_rl.tasks.walking_direct.rewards import (
    compute_walking_terms, compute_feet_air_time_rate, compute_optional_terms, diagonal_trot_cost)
from grallator_rl.tasks.walking_direct.gait_state import GaitState, contact_from_forces
from grallator_rl.him.history import HIMHistory
from grallator_rl.him.walking_checkpoint import verify_warm_start, WARM_START

torch.set_num_threads(2)


class RewardTests(unittest.TestCase):
    def setUp(self):
        self.cfg = WalkingRewardCfg()
        self.g = GaitState(2, 'cpu')
        self.contact = torch.ones(2, 4, dtype=torch.bool)
        self.eligible = torch.ones(2, dtype=torch.bool)

    def step(self, contact=None, eligible=None):
        return self.g.update(self.contact if contact is None else contact,
                             self.eligible if eligible is None else eligible, .02, self.cfg)

    def swing(self, steps=5):
        self.step()
        c = self.contact.clone(); c[:, 0] = False
        for _ in range(steps): self.step(c)
        return self.step()['qualified_airtime_s']

    def test_first_contact_does_not_pay(self):
        self.assertEqual(self.step()['qualified_airtime_s'].sum(), 0)

    def test_touchdown_exactly_once_and_units(self):
        bonus = self.swing()
        torch.testing.assert_close(bonus[:, 0], torch.full((2,), .1))
        rate = compute_feet_air_time_rate(bonus, 2., .02)
        torch.testing.assert_close(rate * .02, torch.full((2,), .2))
        self.assertEqual(self.step()['qualified_airtime_s'].sum(), 0)

    def test_minimum_and_maximum_swing(self):
        for steps, expected in ((3, 0.), (4, .08), (17, .34), (18, 0.)):
            self.g.reset([0, 1])
            self.assertAlmostEqual(self.swing(steps)[0, 0].item(), expected, places=5)

    def test_perpetual_air_never_pays(self):
        c=self.contact.clone(); c[:, 0] = False
        self.step()
        for _ in range(30): self.assertEqual(self.step(c)['qualified_airtime_s'].sum(), 0)

    def test_flight_invalidates_entire_swing(self):
        self.step()
        self.step(torch.zeros_like(self.contact))
        c=self.contact.clone(); c[:,0]=False
        for _ in range(5): self.step(c)
        self.assertEqual(self.step()['qualified_airtime_s'].sum(), 0)

    def test_standing_or_unstable_command_invalidates(self):
        self.step()
        c=self.contact.clone(); c[:, 0]=False
        for _ in range(5): self.step(c, torch.zeros(2, dtype=torch.bool))
        self.assertEqual(self.step()['qualified_airtime_s'].sum(), 0)

    def test_per_env_reset_and_no_fake_touchdown(self):
        self.step(); c=self.contact.clone(); c[:, 0]=False
        for _ in range(5): self.step(c)
        self.g.reset([0])
        q=self.step()['qualified_airtime_s']
        self.assertEqual(q[0].sum(), 0); self.assertAlmostEqual(q[1].sum().item(), .1, places=5)

    def test_reset_exclusion_disarms(self):
        self.swing(); self.step(eligible=torch.zeros(2, dtype=torch.bool))
        c=self.contact.clone(); c[:,0]=False
        for _ in range(5): self.step(c)
        self.assertEqual(self.step()['qualified_airtime_s'].sum(), 0)

    def test_force_threshold_shapes_and_pairing(self):
        f=torch.zeros(2,4,3); f[0,:,2]=torch.tensor([0,2,2.01,10])
        self.assertEqual(contact_from_forces(f,2)[0].tolist(), [False,False,True,True])
        with self.assertRaises(ValueError): contact_from_forces(f[:,:,0],2)
        c=torch.tensor([[True,False,False,True],[False,True,True,False],[True,True,False,False]])
        torch.testing.assert_close(diagonal_trot_cost(c), torch.tensor([0.,0.,1.]))

    def inputs(self):
        z3=torch.zeros(2,3); z12=torch.zeros(2,12)
        return dict(rewards_cfg=self.cfg, base_lin_vel_b=z3.clone(),base_ang_vel_b=z3.clone(),
                    projected_gravity_b=torch.tensor([[0.,0.,-1.]]).repeat(2,1),
                    velocity_commands=z3.clone(),base_height=torch.full((2,),.38),
                    target_height=torch.full((2,),.38),joint_vel=z12.clone(),
                    actions=z12.clone(),previous_actions=z12.clone(),undesired_contact=torch.zeros(2))

    def test_baseline_arithmetic_and_zero_weights(self):
        data=self.inputs(); t=compute_walking_terms(**data)
        self.assertEqual(len(t), 9)
        torch.testing.assert_close(sum(t.values()), torch.full((2,),5.5))
        data['base_lin_vel_b'][:,0]=.15
        torch.testing.assert_close(compute_walking_terms(**data)['track_xy'],torch.full((2,),4/math.e))
        data['actions'][:]=1
        self.assertTrue((compute_walking_terms(**data)['action_rate'] < 0).all())
        self.cfg.track_xy=0
        self.assertEqual(compute_walking_terms(**data)['track_xy'].sum(), 0)
        data['actions']=torch.zeros(2,11)
        with self.assertRaises(ValueError): compute_walking_terms(**data)

    def test_optional_terms_zero_and_sign(self):
        kwargs=dict(contact=self.contact,air_time=torch.rand(2,4),contact_time=torch.rand(2,4),
                    foot_velocity_w=torch.ones(2,4,3),ang_vel_b=torch.ones(2,3),
                    vertical_acceleration=torch.ones(2),joint_acceleration=torch.ones(2,12),
                    joint_error=torch.ones(2,12),joint_velocity=torch.ones(2,12),
                    joint_torque=torch.ones(2,12),commands=torch.zeros(2,3))
        terms=compute_optional_terms(self.cfg,**kwargs)
        self.assertTrue(all(t.shape==(2,) and torch.isfinite(t).all() and t.sum()==0 for t in terms.values()))
        for name in terms: setattr(self.cfg,name,1.)
        terms=compute_optional_terms(self.cfg,**kwargs)
        self.assertTrue(all((t<=0).all() for t in terms.values()))
        torch.testing.assert_close(terms['joint_position'],torch.full((2,),-60.))
        kwargs['contact']=torch.zeros_like(self.contact)
        self.assertEqual(compute_optional_terms(self.cfg,**kwargs)['stance_slip'].sum(),0)

    def test_cfg_validation(self):
        self.cfg.validate_weights()
        self.cfg.feet_air_time=float('nan')
        with self.assertRaises(ValueError): self.cfg.validate_weights()

    def test_history_value_order_and_selective_reset(self):
        h=HIMHistory(2,device='cpu'); a=torch.arange(90.).reshape(2,45)
        torch.testing.assert_close(h.push(a).reshape(2,6,45), a[:,None,:].expand(-1,6,-1))
        b=a+100; out=h.push(b).reshape(2,6,45)
        torch.testing.assert_close(out[:,0],b); torch.testing.assert_close(out[:,1],a)
        h.reset([0]); out=h.push(a).reshape(2,6,45)
        torch.testing.assert_close(out[0],a[0].expand(6,-1)); torch.testing.assert_close(out[1,1],b[1])

    def test_checkpoint_provenance(self):
        self.assertEqual(verify_warm_start()['iteration'],0)


def make_algorithm(n=8, steps=6):
    from rsl_rl.models import MLPModel
    from rsl_rl.storage import RolloutStorage
    from grallator_rl.him.rsl_actor import HIMRslActorModel
    from grallator_rl.him.walking_ppo import WalkingPPO
    from grallator_rl.him.walking_ppo_cfg import WalkingPPORunnerCfg
    cfg=WalkingPPORunnerCfg()
    obs=TensorDict({'policy':torch.randn(n,271)*.1,'critic':torch.randn(n,92)*.1},[n])
    groups=cfg.obs_groups
    actor=HIMRslActorModel(obs,groups,'actor',12,distribution_cfg=cfg.actor.distribution_cfg.to_dict())
    critic=MLPModel(obs,groups,'critic',1,hidden_dims=[512,256,128],activation='elu',obs_normalization=True)
    st=RolloutStorage('rl',n,steps,obs,[12],'cpu')
    alg=WalkingPPO(actor,critic,st,device='cpu',num_learning_epochs=2,num_mini_batches=2,
               learning_rate=cfg.algorithm.learning_rate,hio_learning_rate=1e-5,hio_batch_size=128)
    saved=torch.load(WARM_START,map_location='cpu',weights_only=True)
    alg.load(saved,{'actor':True,'critic':True,'optimizer':False,'iteration':False},True)
    return alg,obs


class ConnectivityTests(unittest.TestCase):
    def test_gae_and_terminal_bootstrap_values(self):
        alg,obs=make_algorithm(steps=2)
        alg.storage.values.fill_(2.)
        alg.storage.rewards[0]=1.; alg.storage.rewards[1]=3.
        alg.storage.dones.zero_(); alg.storage.dones[0,0]=1; alg.storage.dones[1,1]=1
        with patch.object(alg.critic,'forward',return_value=torch.full((8,1),2.)):
            alg.compute_returns(obs)
        self.assertAlmostEqual(alg.storage.returns[0,0,0].item(),1.,places=5)
        self.assertAlmostEqual(alg.storage.returns[0,1,0].item(),3.9205,places=5)
        self.assertAlmostEqual(alg.storage.returns[0,2,0].item(),5.78269,places=5)

    def test_timeout_bootstrap_only_applies_to_timeouts(self):
        alg,obs=make_algorithm()
        with torch.no_grad():
            alg.act(obs); values=alg.transition.values[:,0].clone()
            timeout=torch.zeros(8);timeout[0]=1
            alg.process_env_step(obs,torch.ones(8),torch.ones(8),{'time_outs':timeout})
        self.assertAlmostEqual(alg.storage.rewards[0,0,0].item(),1.+.99*values[0].item(),places=5)
        self.assertEqual(alg.storage.rewards[0,1,0].item(),1.)

    def test_real_ppo_hio_updates_frozen_and_active(self):
        for freeze in ('0','1'):
            torch.manual_seed(42)
            alg,obs=make_algorithm()
            actor_before=copy.deepcopy(alg.actor.state_dict()); critic_before=copy.deepcopy(alg.critic.state_dict())
            ppo_ids={id(p) for g in alg.optimizer.param_groups for p in g['params']}
            hio_ids={id(p) for g in alg.hio_trainer.optimizer.param_groups for p in g['params']}
            self.assertFalse(ppo_ids & hio_ids)
            self.assertTrue(all(id(p) in ppo_ids for p in alg.actor.mlp.parameters()))
            self.assertTrue(all(id(p) in ppo_ids for p in alg.critic.parameters()))
            for i in range(6):
                with torch.no_grad():
                    alg.act(obs)
                    nxt=obs.clone(); nxt['policy']+=.01; nxt['critic']+=.01
                    dones=torch.zeros(8); dones[0]=i==2
                    timeouts=torch.zeros(8); timeouts[0]=i==2
                    alg.process_env_step(nxt,torch.rand(8)*.1,dones,{'time_outs':timeouts})
                    obs=nxt
            alg.compute_returns(obs)
            self.assertTrue(torch.isfinite(alg.storage.returns).all())
            # Simulate residual gradients left by a preceding active HIO update.
            # They must not participate in the next PPO actor clipping norm.
            if freeze=='1':
                for param in alg.actor.estimator.parameters(): param.grad=torch.ones_like(param)
            with patch.dict(os.environ,{'GRALLATOR_FREEZE_HIO':freeze,'GRALLATOR_HIO_LR_OVERRIDE':'1e-5'}):
                metrics=alg.update()
            self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
            self.assertFalse(torch.equal(actor_before['mlp.0.weight'],alg.actor.state_dict()['mlp.0.weight']))
            self.assertFalse(torch.equal(critic_before['mlp.0.weight'],alg.critic.state_dict()['mlp.0.weight']))
            self.assertTrue(alg.actor.mlp[0].weight.grad.norm()>0)
            self.assertTrue(alg.critic.mlp[0].weight.grad.norm()>0)
            changed=not torch.equal(actor_before['estimator.encoder.0.weight'],alg.actor.state_dict()['estimator.encoder.0.weight'])
            self.assertEqual(changed,freeze=='0')
            if freeze=='0':
                for module in (alg.actor.estimator.encoder,alg.actor.estimator.target,alg.actor.estimator.prototypes):
                    self.assertTrue(any(p.grad is not None and p.grad.norm()>0 for p in module.parameters()))
                self.assertGreater(metrics['hio_pairs'],0)
                self.assertGreater(metrics['hio_swap_loss'],0)
            else:
                self.assertEqual(metrics['ppo_kl'],metrics['post_hio_kl'])
                self.assertTrue(all(p.grad is None for p in alg.actor.estimator.parameters()))
            saved=alg.save(); other,_=make_algorithm()
            self.assertTrue(other.load(saved,None,True))
            for k,v in alg.actor.state_dict().items(): torch.testing.assert_close(other.actor.state_dict()[k],v,rtol=0,atol=0)
            self.assertEqual(len(other.optimizer.state),len(alg.optimizer.state))
            self.assertEqual(len(other.hio_trainer.optimizer.state),len(alg.hio_trainer.optimizer.state))
            print('CONNECTIVITY',freeze,metrics,flush=True)


if __name__=='__main__': unittest.main(verbosity=2)
