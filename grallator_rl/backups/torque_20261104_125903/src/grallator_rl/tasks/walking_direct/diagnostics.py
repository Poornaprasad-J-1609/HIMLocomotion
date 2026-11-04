"""Tensor-only snapshots and eligible episode aggregates; no reward effects."""
import torch


class WalkingDiagnostics:
    def __init__(self, num_envs, device):
        self.quality_sum = torch.zeros(num_envs, device=device)
        self.quality_count = torch.zeros_like(self.quality_sum)
        self.air_steps = torch.zeros_like(self.quality_sum)
        self.foot_steps = torch.zeros_like(self.quality_sum)
        self.liftoffs = torch.zeros_like(self.quality_sum)

    def reset(self, ids, log):
        log['Walking/Tracking/episode_xy_error_mps'] = (
            self.quality_sum[ids].sum() / self.quality_count[ids].sum().clamp_min(1))
        log['Walking/Tracking/episode_valid_samples'] = self.quality_count[ids].sum()
        for value in (self.quality_sum, self.quality_count, self.air_steps, self.foot_steps, self.liftoffs):
            value[ids] = 0

    def snapshot(self, *, terms, commands, lin_vel, ang_vel, eligible, contact,
                 events, foot_velocity, height_error, undesired, actions, fallen):
        error_xy = (lin_vel[:, :2] - commands[:, :2]).norm(dim=-1)
        vx_error = (lin_vel[:, 0] - commands[:, 0]).abs()
        self.quality_sum += error_xy * eligible
        self.quality_count += eligible
        self.air_steps += ((~contact) & eligible[:, None]).float().sum(-1)
        self.foot_steps += 4 * eligible
        self.liftoffs += events['liftoffs'].float().sum(-1)
        count = eligible.sum().clamp_min(1)
        slip = foot_velocity[..., :2].norm(dim=-1) * contact
        log = {f'Reward_Rate/{name}': value.detach().mean() for name, value in terms.items()}
        log['Reward_Rate/total'] = torch.stack(list(terms.values())).sum(0).detach().mean()
        for i, name in enumerate(('vx_mps', 'vy_mps', 'yaw_radps')):
            log[f'Walking/Commands/{name}'] = commands[:, i].mean()
            log[f'Diagnostics/command_{name}'] = commands[:, i].mean()
        for i, name in enumerate(('vx_mps', 'vy_mps')):
            log[f'Walking/Tracking/actual_{name}'] = lin_vel[:, i].mean()
            log[f'Diagnostics/actual_{name}'] = lin_vel[:, i].mean()
        log.update({
            'Walking/Tracking/actual_yaw_radps': ang_vel[:, 2].mean(),
            'Walking/Tracking/vx_mae_mps': (vx_error * eligible).sum() / count,
            'Walking/Tracking/xy_error_mps': (error_xy * eligible).sum() / count,
            'Walking/Tracking/raw_xy_error_mps': error_xy.mean(),
            'Walking/Tracking/valid_fraction': eligible.float().mean(),
            'Walking/Gait/contact_count': contact.float().sum(-1).mean(),
            'Walking/Gait/airborne_pct': (~contact).float().mean() * 100,
            'Walking/Gait/liftoffs': events['liftoffs'].float().sum(-1).mean(),
            'Walking/Gait/touchdowns': events['touchdowns'].float().sum(-1).mean(),
            'Walking/Gait/qualified_airtime_s': events['qualified_airtime_s'].sum(-1).mean(),
            'Walking/Gait/stance_slip_mps': slip.sum() / contact.sum().clamp_min(1),
            'Walking/Stability/height_error_cm': height_error.abs().mean() * 100,
            'Walking/Stability/undesired_contact_pct': undesired.float().mean() * 100,
            'Walking/Stability/action_near_limit_pct': (actions.abs() > .95).float().mean() * 100,
            'Walking/Stability/fall_fraction': fallen.float().mean(),
            'Diagnostics/actual_yaw_radps': ang_vel[:, 2].mean(),
            'Diagnostics/vx_mae_mps': vx_error.mean(),
            'Diagnostics/vy_mae_mps': (lin_vel[:, 1] - commands[:, 1]).abs().mean(),
            'Diagnostics/xy_vel_error_mps': error_xy.mean(),
            'Diagnostics/yaw_error_radps': (ang_vel[:, 2] - commands[:, 2]).abs().mean(),
            'Diagnostics/command_speed_mps': commands[:, :2].norm(dim=-1).mean(),
            'Diagnostics/actual_speed_mps': lin_vel[:, :2].norm(dim=-1).mean(),
            'Diagnostics/moving_xy_error_mps': (error_xy * eligible).sum() / count,
            'Diagnostics/moving_command_pct': eligible.float().mean() * 100,
            'Diagnostics/height_error_cm': height_error.abs().mean() * 100,
            'Diagnostics/action_near_limit_pct': (actions.abs() > .95).float().mean() * 100,
            'Diagnostics/undesired_contact_pct': undesired.float().mean() * 100,
            'Diagnostics/feet_contact_count': contact.float().sum(-1).mean(),
            'Diagnostics/all_feet_contact_pct': contact.all(-1).float().mean() * 100,
            'Diagnostics/airborne_pct_cumulative': 100 * self.air_steps.sum() / self.foot_steps.sum().clamp_min(1),
            'Diagnostics/liftoffs_per_1000_foot_steps': 1000 * self.liftoffs.sum() / self.foot_steps.sum().clamp_min(1),
        })
        # Fresh dictionary and scalar tensors each step: RSL-RL retains references
        # until its iteration logger runs. Never mutate old snapshots in place.
        return log
