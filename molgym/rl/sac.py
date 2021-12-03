from typing import List

import torch
from torch.optim import Optimizer

from .agent import AbstractActorCritic


def compute_loss_q(ac, ac_target, batch, gamma, alpha, device):
    pass


def compute_loss_pi(ac, batch, alpha):
    pass


def train(
    ac: AbstractActorCritic,
    ac_target: AbstractActorCritic,
    q_optimizer: Optimizer,
    pi_optimizer: Optimizer,
    data: List[dict],
    gamma: float,
    alpha: float,
    polyak: float,
    device: torch.device,
) -> None:
    """Updates the actor-critic."""

    for batch in data:
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()

        loss_q, q_info = compute_loss_q(ac, ac_target, batch, gamma, alpha, device)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network(s), so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        ac.freeze_q()

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(ac, batch, alpha)
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
