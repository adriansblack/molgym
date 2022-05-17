from typing import List, Optional

import torch

from molgym import data, tools
from .environment import EnvironmentCollection


def rollout(
    agent: torch.nn.Module,
    envs: EnvironmentCollection,
    num_steps: Optional[int],
    num_episodes: Optional[int],  # can be more, in practice
    d_max: float,
    batch_size: int,
    device: torch.device,
    training=False,
) -> List[data.Trajectory]:
    assert num_steps is not None or num_episodes is not None

    num_iters = num_steps // len(envs) if num_steps is not None else None
    iter_counter = 0
    episode_counter = 0

    states = envs.reset_all()
    unfinished_taus: List[List[data.SARS]] = [[] for _ in range(len(envs))]
    taus: List[data.Trajectory] = []

    while ((num_iters is None or iter_counter < num_iters)
           and (num_episodes is None or episode_counter < num_episodes)):
        data_loader = data.DataLoader(
            dataset=[data.process_sa(state=state, cutoff=d_max, action=None, infbag=agent.infbag) for state in states],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        action_list: List[tools.TensorDict] = []
        for batch in data_loader:
            batch = tools.dict_to_device(batch, device)
            response, _info = agent(state=batch['state'], action=None, training=training)
            action_list.append(response['action'])

        action_td = tools.concat_tensor_dicts(action_list)
        actions = data.actions_from_td(action_td)

        tuples = envs.step(actions)
        next_states, rewards, dones, _infos = zip(*tuples)
        next_states = list(next_states)

        for tau, s, a, r, next_s, d in zip(unfinished_taus, states, actions, rewards, next_states, dones):
            tau.append(data.SARS(s, a, r, next_s, d))

        for i in range(len(envs)):
            # possible that we end up with more trajectories than num_episodes
            if unfinished_taus[i][-1].done:
                taus.append(unfinished_taus[i])
                episode_counter += 1
                unfinished_taus[i] = []
                next_states[i] = envs.reset_env(i)

        states = next_states
        iter_counter += 1

    return taus
