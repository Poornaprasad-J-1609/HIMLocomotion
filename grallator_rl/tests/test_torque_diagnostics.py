"""CPU tests of the actual Warp collector and native RSL-RL event routing."""
import tempfile
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import warp as wp

from grallator_rl.tasks.walking_direct.torque_diagnostics import (
    JOINT_NAMES, TorqueAccumulator, joint_indices)


class TorqueTests(unittest.TestCase):
    def test_monitoring_does_not_change_policy_gradients_or_checkpoint(self):
        helper=Path(__file__).with_name('test_native_walking.py')
        spec=importlib.util.spec_from_file_location('torque_test_ppo_helper',helper)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        torch.manual_seed(42);base,obs=module.make_algorithm()
        torch.manual_seed(42);monitored,_=module.make_algorithm()
        accumulator=TorqueAccumulator(8,torch.full((8,12),100.))
        self.sample(accumulator,torch.ones(8,12))
        from grallator_rl.tasks.walking_direct.diagnostics import WalkingDiagnostics
        diagnostics=WalkingDiagnostics(8,'cpu')
        diagnostics.torque=SimpleNamespace(accumulator=accumulator)
        monitored._torque_env=SimpleNamespace(diagnostics=diagnostics,extras={'log':{}})
        outcomes=[]
        for algorithm in (base,monitored):
            torch.manual_seed(123)
            for _ in range(6):
                with torch.no_grad():
                    algorithm.act(obs)
                    algorithm.process_env_step(obs,torch.arange(8)*.01,torch.zeros(8),{})
            algorithm.compute_returns(obs)
            with patch.dict(os.environ,{'GRALLATOR_FREEZE_HIO':'0','GRALLATOR_HIO_LR_OVERRIDE':'1e-5'}):
                outcomes.append(algorithm.update())
        self.assertEqual(outcomes[0],outcomes[1])
        for original,logged in ((base.actor,monitored.actor),(base.critic,monitored.critic)):
            for name,value in original.state_dict().items():
                torch.testing.assert_close(value,logged.state_dict()[name],atol=0,rtol=0)
            for first,second in zip(original.parameters(),logged.parameters()):
                if first.grad is not None:torch.testing.assert_close(first.grad,second.grad,atol=0,rtol=0)
        self.assertEqual(base.save().keys(),monitored.save().keys())
        self.assertIn('Torque/OverallPeakAbs_Nm',monitored._torque_env.extras['log'])

    def make(self, capacity=8):
        return TorqueAccumulator(2, torch.full((2, 12), 10.), capacity=capacity)

    def sample(self, accumulator, value, columns=None):
        value=value.float().contiguous()
        columns=wp.array(list(range(12)) if columns is None else columns, dtype=wp.int32,device='cpu')
        accumulator.sample(wp.from_torch(value), columns, wp.from_torch(value * 2), columns)
        wp.synchronize()

    def test_actual_names_exact_mapping(self):
        self.assertEqual(joint_indices(list(reversed(JOINT_NAMES))), list(reversed(range(12))))
        for invalid in (list(JOINT_NAMES)[:-1],list(JOINT_NAMES)+['other'],[JOINT_NAMES[0]]*12):
            with self.assertRaises(ValueError): joint_indices(invalid)

    def test_signed_peaks_mean_p95_fraction_and_time_accumulation(self):
        a=self.make()
        x=torch.arange(24.).reshape(2,12)-12
        y=-torch.arange(24.).reshape(2,12)/2
        columns=list(reversed(range(12)))
        self.sample(a,x,columns);self.sample(a,y,columns)
        expected=torch.cat((x[:,columns].abs(),y[:,columns].abs()))
        metrics=a.finish_interval()
        for j,name in enumerate(JOINT_NAMES):
            self.assertAlmostEqual(metrics[f'Torque/PeakAbs_Nm/{name}'].item(),expected[:,j].max().item())
            self.assertAlmostEqual(metrics[f'Torque/MeanAbs_Nm/{name}'].item(),expected[:,j].mean().item())
            self.assertAlmostEqual(metrics[f'Torque/P95Abs_Nm/{name}'].item(),torch.quantile(expected[:,j],.95).item(),places=5)
            self.assertEqual(metrics[f'Torque/EffortLimitFraction/{name}'],(expected[:,j]>=9.5).float().mean())
        self.assertEqual(metrics['Torque/OverallPeakAbs_Nm'],12.)
        self.assertEqual(metrics['Torque/PhysicsSamplesPerJoint'],4.)
        self.assertEqual(metrics[f'EstimatedTorque/UnclippedComputedPeakAbs_Nm/{JOINT_NAMES[0]}'],23.)

    def test_interval_and_run_boundaries_and_immutable_snapshots(self):
        a=self.make();self.sample(a,torch.full((2,12),-7.));old=a.finish_interval()
        self.sample(a,torch.ones(2,12));new=a.finish_interval()
        self.assertEqual(new['Torque/OverallPeakAbs_Nm'],1.)
        self.assertEqual(new['Torque/OverallMaxEver_Nm'],7.)
        self.assertEqual(old['Torque/OverallPeakAbs_Nm'],7.)
        a.reset_run();empty=a.finish_interval()
        self.assertEqual(empty['Torque/OverallMaxEver_Nm'],0.)
        self.assertEqual(empty['Torque/PhysicsSamplesPerJoint'],0.)

    def test_nonfinite_and_overflow_fail_at_boundary(self):
        a=self.make();x=torch.ones(2,12);x[0,0]=float('nan');self.sample(a,x)
        with self.assertRaises(RuntimeError):a.finish_interval()
        a=self.make(capacity=1);self.sample(a,torch.ones(2,12));self.sample(a,torch.ones(2,12))
        with self.assertRaises(RuntimeError):a.finish_interval()

    def test_sampled_peak_labels_when_decimation_prevents_substeps(self):
        a=TorqueAccumulator(2,torch.ones(2,12),sample_dt=.02,sampled_peak=True)
        metrics=a.finish_interval()
        self.assertIn('Torque/SampledPeakAbs_Nm/'+JOINT_NAMES[0],metrics)
        self.assertNotIn('Torque/PeakAbs_Nm/'+JOINT_NAMES[0],metrics)

    def test_native_logger_exact_peak_not_mean_of_step_maxima(self):
        from rsl_rl.utils.logger import Logger
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        from grallator_rl.tasks.walking_direct.diagnostics import WalkingDiagnostics
        a=self.make();self.sample(a,torch.ones(2,12));self.sample(a,torch.full((2,12),-9.))
        with tempfile.TemporaryDirectory() as path:
            logger=Logger(path,{'algorithm':{},'num_steps_per_env':2,'logger':'tensorboard'}, {},2,False,1,0,'cpu')
            logger.init_logging_writer()
            first={'Reward_Rate/total':torch.tensor(1.)};last={'Reward_Rate/total':torch.tensor(2.)}
            for snapshot in (first,last):logger.process_env_step(torch.ones(2),torch.zeros(2),{'log':snapshot})
            diagnostics=WalkingDiagnostics(2,'cpu')
            class Adapter:accumulator=a
            diagnostics.torque=Adapter()
            diagnostics.reset([0],{})  # episode reset must not lose interval peaks
            diagnostics.publish_torque_iteration(last)
            self.assertFalse(any(key.startswith('Torque/') for key in first))
            logger.log(0,0,1,.01,.01,{},1e-5,torch.ones(12),None,print_minimal=True)
            logger.stop_logging_writer()
            events=EventAccumulator(path).Reload()
            expected={f'Torque/{group}/{joint}' for group in
                      ('PeakAbs_Nm','MaxEver_Nm','MeanAbs_Nm','P95Abs_Nm','EffortLimitFraction') for joint in JOINT_NAMES}
            expected|={'Torque/OverallPeakAbs_Nm','Torque/OverallMaxEver_Nm'}
            self.assertTrue(expected.issubset(events.Tags()['scalars']))
            self.assertEqual(events.Scalars('Torque/OverallPeakAbs_Nm')[0].value,9.)
            self.assertEqual(events.Scalars('Torque/MeanAbs_Nm/'+JOINT_NAMES[0])[0].value,5.)


if __name__=='__main__':unittest.main(verbosity=2)
