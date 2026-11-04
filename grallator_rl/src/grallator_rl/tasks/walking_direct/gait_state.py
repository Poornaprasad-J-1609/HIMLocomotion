"""Control-sampled contact transitions; no module-global reward state.

The sensor's current normal-force buffer is sampled once per control interval
(0.02 s). Substep-only transitions are intentionally not reconstructed. A swing
must start from observed support and remain supported/stable/moving throughout.
"""
import math
import torch

FOOT_NAMES = ('FR_foot', 'FL_foot', 'BR_foot', 'BL_foot')


def contact_from_forces(forces, threshold):
    if forces.ndim != 3 or forces.shape[1:] != (4, 3):
        raise ValueError('Expected [N,4,3] foot normal forces')
    return forces.norm(dim=-1) > threshold


class GaitState:
    def __init__(self, num_envs, device):
        shape = (num_envs, 4)
        self.contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.seen_contact = torch.zeros_like(self.contact)
        self.valid_swing = torch.zeros_like(self.contact)
        self.air_time = torch.zeros(shape, device=device)
        self.contact_time = torch.zeros_like(self.air_time)
        self.last_air_time = torch.zeros_like(self.air_time)
        self.last_contact_time = torch.zeros_like(self.air_time)

    def reset(self, ids):
        for tensor in (self.contact, self.seen_contact, self.valid_swing,
                       self.air_time, self.contact_time, self.last_air_time,
                       self.last_contact_time):
            tensor[ids] = 0

    def update(self, contact, eligible, dt, cfg):
        if contact.shape != self.contact.shape or eligible.shape != contact.shape[:1]:
            raise ValueError('Unexpected gait-state shapes')
        if contact.dtype != torch.bool or eligible.dtype != torch.bool or not math.isfinite(dt) or dt <= 0:
            raise ValueError('Expected boolean contact/gate and positive finite dt')
        supported = contact.sum(-1) >= cfg.min_support_feet
        gate = eligible & supported
        liftoff = self.contact & ~contact & self.seen_contact
        touchdown = ~self.contact & contact & self.seen_contact
        self.valid_swing = torch.where(liftoff, gate[:, None], self.valid_swing)
        self.valid_swing &= gate[:, None]
        # air_time includes the first observed airborne interval, excludes the
        # touchdown interval. Bounds apply to the actual completed swing.
        valid = (touchdown & self.valid_swing
                 & (self.air_time >= cfg.min_swing_s - 1e-6)
                 & (self.air_time <= cfg.max_swing_s + 1e-6))
        qualified = torch.where(valid, self.air_time.clamp(max=cfg.max_swing_s), 0.0)
        self.last_air_time = torch.where(valid, self.air_time, self.last_air_time)
        self.last_contact_time = torch.where(liftoff & gate[:, None], self.contact_time, self.last_contact_time)
        self.air_time = torch.where(~contact & eligible[:, None], self.air_time + dt, 0.0)
        self.contact_time = torch.where(contact & eligible[:, None], self.contact_time + dt, 0.0)
        self.valid_swing &= ~contact
        # Only contacts outside reset transients can arm a subsequent liftoff.
        self.seen_contact |= contact & eligible[:, None]
        self.seen_contact &= eligible[:, None]
        self.contact.copy_(contact)
        return {'qualified_airtime_s': qualified, 'liftoffs': liftoff & eligible[:, None],
                'touchdowns': touchdown & eligible[:, None]}
