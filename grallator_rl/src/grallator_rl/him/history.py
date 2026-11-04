"""Unchanged HIM observation-history structure: 6 frames x 45 values."""
import torch


class HIMHistory:
    def __init__(self, num_envs, obs_dim=45, history_steps=6, device="cuda:0"):
        self.num_envs = num_envs
        self.obs_dim = obs_dim
        self.history_steps = history_steps
        self.buffer = torch.zeros(num_envs, history_steps, obs_dim, device=device)
        self.needs_fill = torch.ones(num_envs, dtype=torch.bool, device=device)

    @torch.no_grad()
    def reset(self, env_ids):
        self.buffer[env_ids] = 0.0
        self.needs_fill[env_ids] = True

    @torch.no_grad()
    def push(self, obs):
        expected = (self.num_envs, self.obs_dim)
        if obs.shape != expected:
            raise ValueError(f"Expected observation {expected}, got {tuple(obs.shape)}")

        self.buffer[:, 1:, :].copy_(self.buffer[:, :-1, :].clone())
        self.buffer[:, 0, :].copy_(obs)
        self.buffer.copy_(
            torch.where(self.needs_fill[:, None, None], obs[:, None, :], self.buffer)
        )
        self.needs_fill.zero_()
        return self.buffer.flatten(start_dim=1).clone()
