import logging
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
        response, _aux = ac.policy(batch['next_state'], action=None, training=True)
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

    loss_q1 = torch.square(q1 - backup).mean()
    loss_q2 = torch.square(q2 - backup).mean()
    loss = loss_q1 + loss_q2

    return loss


def compute_surrogate_loss_policy(
    ac: SACAgent,
    batch: Dict,
    alpha: float,
    cutoff: float,
    device: torch.device,
) -> torch.Tensor:
    response, _aux = ac.policy(batch['state'], action=None, training=True)
    s_next = data.propagate_batch(batch['state'], response['action'], cutoff=cutoff)
    s_next.to(device)

    q1 = ac.q1(s_next)  # [B, ]
    q2 = ac.q2(s_next)  # [B, ]
    q = torch.minimum(q1, q2)

    # Entropy-regularized policy loss surrogate
    # q_pi.detach() just to be sure
    loss = ((alpha * response['logp'].detach() + alpha - q.detach()) * response['logp']).mean()

    return loss


def train_epoch(
    ac: SACAgent,
    ac_target: SACTarget,
    optimizer: Optimizer,
    data_loader: data.DataLoader,
    alpha: float,
    polyak: float,
    cutoff: float,  # Angstrom
    device: torch.device,
) -> Dict[str, Any]:
    """Updates the actor-critic."""

    infos: List[tools.TensorDict] = []

    for batch in data_loader:
        batch_info = {}
        batch = tools.dict_to_device(batch, device)

        optimizer.zero_grad()

        # First run one gradient descent step for Q1 and Q2
        loss_q = compute_loss_q(ac, ac_target, batch, alpha=alpha, cutoff=cutoff, device=device)

        # Freeze Q-network(s), so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        ac.freeze_q()

        # Next run one gradient descent step for pi.
        loss_pi = compute_surrogate_loss_policy(ac, batch, alpha=alpha, cutoff=cutoff, device=device)

        # Unfreeze Q-network(s), so you can optimize it at next step.
        ac.unfreeze_q()

        loss = loss_pi + loss_q
        loss.backward()
        optimizer.step()

        batch_info['loss_pi'] = loss_pi.detach().cpu()
        batch_info['loss_q'] = loss_q.detach().cpu()
        batch_info['loss'] = loss.detach().cpu()

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
    return info


def train(
    ac: SACAgent,
    ac_target: SACTarget,
    optimizer: Optimizer,
    data_loader: data.DataLoader,
    alpha: float,
    polyak: float,
    cutoff: float,  # Angstrom
    num_epochs: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    logging.debug(f'Training for {num_epochs} epoch(s)')
    info = []
    for epoch in range(num_epochs):
        metrics = train_epoch(
            ac=ac,
            ac_target=ac_target,
            optimizer=optimizer,
            data_loader=data_loader,
            alpha=alpha,
            polyak=polyak,
            cutoff=cutoff,
            device=device,
        )
        metrics['epoch'] = epoch
        info.append(metrics)

    logging.debug('Training complete')
    return info
