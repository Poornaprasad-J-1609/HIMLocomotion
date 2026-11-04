import torch
import torch.nn as nn
POLICY_OBS_DIM = 45
CRITIC_OBS_DIM = 91


@torch.no_grad()
def build_critic_observation(
    robot,
    policy_obs,
    foot_body_ids,
    env_origins,
):
    num_envs = policy_obs.shape[0]

    if policy_obs.shape != (num_envs, POLICY_OBS_DIM):
        raise ValueError("Expected 45D policy observations")

    if len(foot_body_ids) != 4:
        raise ValueError("Expected exactly four foot body IDs")

    # True base linear velocity: [N, 3]
    base_lin_vel = robot.data.root_lin_vel_b.torch

    # Base position: [N, 3]
    base_pos = robot.data.root_pos_w.torch

    # Base position relative to its environment origin
    base_pos_local = base_pos - env_origins

    # Four foot positions: [N, 4, 3]
    foot_pos = robot.data.body_link_pos_w.torch[
        :, foot_body_ids, :
    ]

    # Four foot linear velocities: [N, 4, 3]
    foot_vel = robot.data.body_link_lin_vel_w.torch[
        :, foot_body_ids, :
    ]

    # Relative foot positions: [N, 12]
    foot_pos_rel = (
        foot_pos - base_pos.unsqueeze(1)
    ).reshape(num_envs, 12)

    # Flatten foot velocities: [N, 12]
    foot_vel_flat = foot_vel.reshape(num_envs, 12)

    # Foot-center heights relative to environment origins
    foot_heights = (
        foot_pos[:, :, 2]
        - env_origins[:, None, 2]
    )

    # Applied joint efforts: [N, 12]
    joint_efforts = robot.actuators.applied_effort.torch

    if joint_efforts.shape != (num_envs, 12):
        raise ValueError(
            f"Unexpected effort shape: {joint_efforts.shape}"
        )

    critic_obs = torch.cat(
        [
            policy_obs,
            base_lin_vel,
            base_pos_local,
            foot_pos_rel,
            foot_vel_flat,
            foot_heights,
            joint_efforts,
        ],
        dim=-1,
    )

    assert critic_obs.shape == (
        num_envs, CRITIC_OBS_DIM
    )

    return critic_obs

class HIMCritic(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(CRITIC_OBS_DIM, 512),
            nn.ELU(),

            nn.Linear(512, 256),
            nn.ELU(),

            nn.Linear(256, 128),
            nn.ELU(),

            nn.Linear(128, 1),
        )

    def forward(self, critic_obs):

        if (
            critic_obs.ndim != 2
            or critic_obs.shape[1] != CRITIC_OBS_DIM
        ):
            raise ValueError(
                f"Expected critic input [N,91], "
                f"got {tuple(critic_obs.shape)}"
            )

        return self.network(critic_obs)