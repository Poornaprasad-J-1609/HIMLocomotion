import torch
import torch.nn.functional as F


@torch.no_grad()
def sinkhorn_assignments(
    scores,
    epsilon=0.05,
    iterations=3,
):
    # scores: [batch_size, 32 prototypes]

    q = torch.exp(
        (scores.float() - scores.float().max()) / epsilon
    ).T

    q /= q.sum().clamp_min(1e-12)

    num_prototypes, batch_size = q.shape

    for _ in range(iterations):

        # Normalize across prototypes
        q /= q.sum(dim=1, keepdim=True).clamp_min(1e-12)
        q /= num_prototypes

        # Normalize across samples
        q /= q.sum(dim=0, keepdim=True).clamp_min(1e-12)
        q /= batch_size

    return (q * batch_size).T


def compute_hio_losses(
    estimator,
    history_t,
    next_obs,
    next_true_velocity,
):
    batch_size = history_t.shape[0]

    assert history_t.shape == (batch_size, 270)
    assert next_obs.shape == (batch_size, 45)
    assert next_true_velocity.shape == (batch_size, 3)

    # Source encoder: history -> velocity + latent
    predicted_velocity, source_z = estimator.encode(
        history_t.detach()
    )

    # Our 45D observation order:
    # 0:3   angular velocity
    # 3:6   projected gravity
    # 6:9   velocity commands
    # 9:45  joint state and previous action
    #
    # Target input: 42D proprioception + 3D true velocity
    target_input = torch.cat(
        (
            next_obs[:, :6],
            next_obs[:, 9:],
            next_true_velocity,
        ),
        dim=-1,
    )

    # Target encoder: successor information -> 16D latent
    target_z = F.normalize( 
        estimator.target(target_input.detach()),
        p=2,
        dim=-1,
        eps=1e-8,
    )

    # Normalize prototype vectors
    with torch.no_grad():
        estimator.prototypes.weight.copy_(
            F.normalize(
                estimator.prototypes.weight,
                p=2,
                dim=1,
                eps=1e-8,
            )
        )

    # Compare latent vectors with learnable prototypes
    source_scores = (
        source_z @ estimator.prototypes.weight.T
    )

    target_scores = (
        target_z @ estimator.prototypes.weight.T
    )

    # Balanced prototype assignments
    source_assignments = sinkhorn_assignments(source_scores)
    target_assignments = sinkhorn_assignments(target_scores)

    # Predicted prototype probabilities
    source_log_probs = F.log_softmax(
        source_scores / estimator.temperature,
        dim=-1,
    )

    target_log_probs = F.log_softmax(
        target_scores / estimator.temperature,
        dim=-1,
        
    )

    # Swapped-assignment contrastive objective
    swap_loss = -0.5 * (
        source_assignments * target_log_probs
        + target_assignments * source_log_probs
    ).mean()

    # Explicit velocity-estimation objective
    velocity_loss = F.mse_loss(
        predicted_velocity,
        next_true_velocity.detach(),
    )

    return velocity_loss, swap_loss     