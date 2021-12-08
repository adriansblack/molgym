import time
from typing import Dict, List, Any

import torch
from torch.optim import Optimizer

from molgym import tools, data
from .agent import SACAgent, SACTarget


def compute_loss_q(
    ac: SACAgent,
    ac_target: SACTarget,
    batch: Dict,
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
        backup = batch['reward'] + (1 - batch['done']) * (q_target - alpha * response['logp'])

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
    s_next = data.propagate_batch(batch['state'], actions, cutoff=cutoff)
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
    alpha: float,
    polyak: float,
    cutoff: float,  # Angstrom
    device: torch.device,
) -> Dict[str, Any]:
    """Updates the actor-critic."""

    start_time = time.time()
    infos: List[tools.TensorDict] = []

    for batch in data_loader:
        batch_info = {}
        batch = tools.dict_to_device(batch, device)

        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(ac, ac_target, batch, alpha=alpha, cutoff=cutoff, device=device)
        loss_q.backward()
        q_optimizer.step()
        batch_info['loss_q'] = loss_q.detach()

        # Freeze Q-network(s), so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        ac.freeze_q()

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi = compute_surrogate_loss_policy(ac, batch, alpha=alpha, cutoff=cutoff, device=device)
        loss_pi.backward()
        pi_optimizer.step()
        batch_info['loss_pi'] = loss_pi.detach()

        # Unfreeze Q-network(s), so you can optimize it at next step.
        ac.unfreeze_q()

        batch_info['loss'] = (loss_q + loss_pi).detach()

        # Finally, update target networks by Polyak averaging.
        with torch.no_grad():
            # NB: We use an in-place operations "mul_", "add_" to update target
            # params, as opposed to "mul" and "add", which would make new tensors.
            for p, p_target in zip(ac.q1.parameters(), ac_target.q1.parameters()):
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)

            for p, p_target in zip(ac.q2.parameters(), ac_target.q2.parameters()):
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)

        infos.append(batch_info)

    merged_info = tools.stack_tensor_dicts(infos)
    means = tools.apply_to_dict(merged_info, torch.mean, axis=0)
    info = tools.apply_to_dict(means, tools.to_numpy)
    info['time'] = time.time() - start_time
    return info
