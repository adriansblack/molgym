from typing import Dict

import torch
from torch.optim import Optimizer

from molgym import tools, data
from .agent import SACAgent


def propagate_batch(states: data.StateData, actions: tools.TensorDict, cutoff: float) -> data.StateData:
    s_list = [data.state_from_td(state) for state in states.to_data_list()]
    a_list = data.actions_from_td(actions)
    s_next_list = [data.propagate(s, a) for s, a in zip(s_list, a_list)]
    s_next_processed = [data.process_sa(s_next, cutoff=cutoff) for s_next in s_next_list]
    return data.collate_fn(s_next_processed)


def compute_loss_q(
        ac: SACAgent,
        ac_target: SACAgent,
        batch: Dict,
        gamma: float,
        alpha: float,
        cutoff: float,  # Angstrom
) -> torch.Tensor:
    q1 = ac.q1(batch['next_state'])
    q2 = ac.q2(batch['next_state'])

    # Bellman backup for Q functions
    with torch.no_grad():
        # Target actions come from *current* policy
        response, _aux = ac.policy(batch['next_state'])
        s_next_next = propagate_batch(batch['next_state'], response['action'], cutoff=cutoff)

        # Target Q-values
        q1_pi_target = ac_target.q1(s_next_next)  # [n_states, ]
        q2_pi_target = ac_target.q2(s_next_next)  # [n_states, ]
        q_pi_target = torch.minimum(q1_pi_target, q2_pi_target)
        backup = batch['reward'] + gamma * (1 - batch['done']) * (q_pi_target - alpha * response['logp'])

    # MSE loss against Bellman backup
    loss_q1 = ((q1 - backup)**2).mean()
    loss_q2 = ((q2 - backup)**2).mean()
    loss_q = loss_q1 + loss_q2

    return loss_q


def compute_loss_pi(ac: SACAgent, batch: Dict, alpha: float, cutoff: float) -> torch.Tensor:
    response, _aux = ac.policy(batch['state'])
    actions = tools.detach_tensor_dict(response['action'])  # don't take gradient through samples
    s_next = propagate_batch(batch['next_state'], actions, cutoff=cutoff)

    q1_pi = ac.q1(s_next)
    q2_pi = ac.q2(s_next)

    q_pi = torch.minimum(q1_pi, q2_pi)

    # Entropy-regularized policy loss
    loss_pi = ((alpha * response['logp'].detach() + alpha - q_pi) * response['logp']).mean()

    return loss_pi


def train(
    ac: SACAgent,
    ac_target: SACAgent,
    q_optimizer: Optimizer,
    pi_optimizer: Optimizer,
    data_loader: data.DataLoader,
    gamma: float,
    alpha: float,
    polyak: float,
    cutoff: float,
    device: torch.device,
) -> None:
    """Updates the actor-critic."""

    for batch in data_loader:
        batch = tools.dict_to_device(batch, device)

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(ac, ac_target, batch, gamma, alpha, cutoff=cutoff)
        loss_q.backward()
        q_optimizer.step()

        # Freeze Q-network(s), so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        ac.freeze_q()

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(ac, batch, alpha, cutoff=cutoff)
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
