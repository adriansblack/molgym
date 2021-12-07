from typing import Dict

import torch
from torch.optim import Optimizer

from molgym import tools, data
from .agent import SACAgent, SACTarget


def compute_loss_q(
    ac: SACAgent,
    ac_target: SACTarget,
    batch: Dict,
    gamma: float,
    alpha: float,
    cutoff: float,  # Angstrom
    device: torch.device,
) -> torch.Tensor:

    # Bellman backup for Q functions
    with torch.no_grad():
        # Target actions come from *current* policy
        response, _aux = ac.policy(batch['next_state'])
        s_next_next = data.propagate_batch(batch['next_state'], response['action'], cutoff=cutoff)
        s_next_next.to(device)

        # Target Q-values
        q1_target = ac_target.q1(s_next_next)  # [B, ]
        q2_target = ac_target.q2(s_next_next)  # [B, ]
        q_target = torch.minimum(q1_target, q2_target)
        backup = batch['reward'] + gamma * (1 - batch['done']) * (q_target - alpha * response['logp'])

    # MSE loss against Bellman backup
    q1 = ac.q1(batch['next_state'])
    q2 = ac.q2(batch['next_state'])

    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss = loss_q1 + loss_q2

    return loss


def compute_surrogate_loss_policy(
    ac: SACAgent,
    batch: Dict,
    alpha: float,
    cutoff: float,
    device: torch.device,
) -> torch.Tensor:
    response, _aux = ac.policy(batch['state'])
    actions = tools.detach_tensor_dict(response['action'])  # don't take gradient through samples
    s_next = data.propagate_batch(batch['next_state'], actions, cutoff=cutoff)
    s_next.to(device)

    q1 = ac.q1(s_next)  # [B, ]
    q2 = ac.q2(s_next)  # [B, ]
    q = torch.minimum(q1, q2)

    # Entropy-regularized policy loss surrogate
    # q_pi.detach() just to be sure
    loss = ((alpha * response['logp'].detach() + alpha - q.detach()) * response['logp']).mean()

    return loss


def train(
    ac: SACAgent,
    ac_target: SACTarget,
    q_optimizer: Optimizer,
    pi_optimizer: Optimizer,
    data_loader: data.DataLoader,
    gamma: float,
    alpha: float,
    polyak: float,
    cutoff: float,  # Angstrom
    device: torch.device,
) -> None:
    """Updates the actor-critic."""

    for batch in data_loader:
        batch = tools.dict_to_device(batch, device)

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(ac, ac_target, batch, gamma=gamma, alpha=alpha, cutoff=cutoff, device=device)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network(s), so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        ac.freeze_q()

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_surrogate_loss_policy(ac, batch, alpha=alpha, cutoff=cutoff, device=device)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-network(s), so you can optimize it at next step.
        ac.unfreeze_q()

        # Finally, update target networks by Polyak averaging.
        with torch.no_grad():
            for p, p_target in zip(ac.parameters(), ac_target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)
