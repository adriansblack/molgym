from typing import List, Optional

import torch
import torch_geometric

from molgym import data, tools
from .environment import EnvironmentCollection


def rollout(
    policy,
    envs: EnvironmentCollection,
    num_steps: Optional[int],
    num_episodes: Optional[int],  # can be more, in practice
    d_max: float,
    batch_size: int,
    device: torch.device,
    training=False,
) -> List[data.SARS]:
    assert num_steps is not None or num_episodes is not None

    num_iters = num_steps // len(envs) if num_steps is not None else None
    iter_counter = 0
    episode_counter = 0

    states = envs.reset_all()
    sars_lists: List[List[data.SARS]] = [[] for _ in range(len(envs))]
    buffer: List[data.SARS] = []

    while ((num_iters is None or iter_counter < num_iters)
           and (num_episodes is None or episode_counter < num_episodes)):
        data_loader = torch_geometric.loader.DataLoader(
            dataset=[data.build_state_action_data(state=state, cutoff=d_max, action=None) for state in states],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )

        responses: List[tools.TensorDict] = []
        for batch in data_loader:
            batch = batch.to(device)
            response, _ = policy(batch, training=training)
            responses.append(response)

        response = tools.concat_tensor_dicts(responses)
        actions = data.get_actions_from_td(response)

        tuples = envs.step(actions)
        next_states, rewards, dones, _infos = zip(*tuples)
        next_states = list(next_states)

        for sars_list, s, a, r, next_s, d in zip(sars_lists, states, actions, rewards, next_states, dones):
            sars_list.append(data.SARS(s, a, r, next_s, d))

        for i, sars_list in enumerate(sars_lists):
            if sars_list[-1].done:
                buffer += sars_list
                sars_list.clear()
                episode_counter += 1
                next_states[i] = envs.reset_env(i)

        states = next_states
        iter_counter += 1

    return buffer
