import torch

OBS_DIM = 45
ACTION_DIM = 12


def build_policy_observation(
    base_ang_vel,
    projected_gravity,
    velocity_commands,
    joint_position_error,
    joint_vel,
    previous_action,
):

    components = [
        ("base_ang_vel", base_ang_vel, 3),
        ("projected_gravity", projected_gravity, 3),
        ("velocity_commands", velocity_commands, 3),
        ("joint_position_error", joint_position_error, 12),
        ("joint_vel", joint_vel, 12),
        ("previous_action", previous_action, 12),
    ]

    for name, tensor, expected_dim in components:
        assert tensor.ndim == 2, f"{name} must be batched"
        assert tensor.shape[1] == expected_dim, name

    obs = torch.cat(
        [tensor for _, tensor, _ in components],
        dim=-1,
    )

    assert obs.shape[1] == OBS_DIM

    return obs