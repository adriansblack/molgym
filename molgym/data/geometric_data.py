from typing import Optional, List, Dict, Union

import numpy as np
import torch.utils.data
import torch_geometric

from molgym import tools
# from molgym.rl.environment import EnvironmentCollection
from . import utils, trajectory, graph_tools
from .trajectory import (State, Action, FOCUS_KEY, ELEMENT_KEY, DISTANCE_KEY, ORIENTATION_KEY, ELEMENTS_KEY,
                         POSITIONS_KEY, BAG_KEY, Bag, propagate)

collate_fn = torch_geometric.loader.dataloader.Collater([], [])


class DataLoader(torch.utils.data.DataLoader):
    """A data loader that merges geometric data objects to a mini-batch."""
    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )


class CanvasData(torch_geometric.data.Data):
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    elements: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor


class CanvasBatch(torch_geometric.data.Batch, CanvasData):
    pass


class StateData(CanvasData):
    bag: torch.Tensor


class StateBatch(torch_geometric.data.Batch, StateData):
    pass


def tensorize_canvas(
    elements: np.ndarray,  # [n, ], not Zs but indices
    positions: np.ndarray,  # [n, 3]
    cutoff: float,
    num_classes: int,
) -> Dict[str, Union[torch.Tensor, int]]:
    assert len(elements.shape) == 1 and len(positions.shape) == 2
    assert elements.shape[0] == positions.shape[0]
    assert positions.shape[1] == 3

    elements_tensor = torch.tensor(elements, dtype=torch.long)
    one_hot_attrs = tools.to_one_hot(indices=elements_tensor.unsqueeze(-1), num_classes=num_classes)
    edge_index, shifts = graph_tools.get_neighborhood(positions=positions, cutoff=cutoff)

    return dict(
        num_nodes=len(elements),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
        node_attrs=one_hot_attrs.to(torch.get_default_dtype()),
        elements=elements_tensor,
        positions=torch.tensor(positions, dtype=torch.get_default_dtype()),
    )


def tensorize_bag(bag: Bag, floats: bool) -> tools.TensorDict:
    if floats: return dict(bag=torch.tensor([bag.flatten()]))
    else: return dict(bag=torch.tensor([bag.flatten()], dtype=torch.long))


def geometrize_config(
    config: utils.Configuration,
    s_table: utils.SymbolTable,
    cutoff: float,
) -> CanvasData:
    elements = np.array([s_table.symbol_to_element(s) for s in config.symbols], dtype=int)
    info = tensorize_canvas(elements, config.positions, cutoff=cutoff, num_classes=len(s_table))
    return CanvasData(**info)


def geometrize_state(state: State, cutoff: float, num_classes: int=None, floats: bool=False) -> StateData:
    if num_classes is None:
        num_classes = len(state.bag) if state.bag.ndim==1 else len(state.bag[0])
    return StateData(
        **tensorize_canvas(elements=state.elements,
                           positions=state.positions,
                           cutoff=cutoff,
                           num_classes=num_classes),
        **tensorize_bag(state.bag, floats),
    )


def tensorize_action(action: Action) -> tools.TensorDict:
    return dict(
        focus=torch.tensor(action.focus, dtype=torch.long),
        element=torch.tensor(action.element, dtype=torch.long),
        distance=torch.tensor(action.distance, dtype=torch.get_default_dtype()),
        orientation=torch.tensor(action.orientation, dtype=torch.get_default_dtype()),
    )


def actions_from_td(td: tools.TensorDict) -> List[Action]:
    return [
        Action(focus=f, element=e, distance=d, orientation=o) for f, e, d, o in zip(
            tools.to_numpy(td[FOCUS_KEY]),
            tools.to_numpy(td[ELEMENT_KEY]),
            tools.to_numpy(td[DISTANCE_KEY]),
            tools.to_numpy(td[ORIENTATION_KEY]),
        )
    ]


def process_sa(
    state: State,
    cutoff: float,
    action: Optional[Action] = None,
    infbag: bool = False
) -> Dict[str, Union[tools.TensorDict, StateData]]:
    info: Dict[str, Union[tools.TensorDict, StateData]] = {'state': geometrize_state(state, cutoff,floats=infbag)}

    if action:
        info['action'] = tensorize_action(action)

    return info


def process_sars(
    sars: trajectory.SARS,
    cutoff: float,
    infbag: bool=False,
) -> Dict[str, Union[torch.Tensor, tools.TensorDict, StateData]]:
    return {
        'state': geometrize_state(sars.state, cutoff=cutoff, floats=infbag),
        'action': tensorize_action(sars.action),
        'reward': torch.tensor(sars.reward, dtype=torch.get_default_dtype()),
        'done': torch.tensor(sars.done, dtype=torch.int),
        'next_state': geometrize_state(sars.next_state, cutoff=cutoff, floats=infbag),
    }


def state_from_td(td: tools.TensorDict) -> State:
    return State(
        elements=tools.to_numpy(td[ELEMENTS_KEY]).reshape(-1),
        positions=tools.to_numpy(td[POSITIONS_KEY]).reshape(-1, 3),
        bag=tools.to_numpy(td[BAG_KEY]).reshape(-1),
    )

def next_bag(byiter: bool, curr_bag: np.ndarray, curr_elements: np.ndarray, bag_schedule: np.ndarray, act_elem: int, stop_idx: int, maskZ: bool = True) -> Bag:
    if byiter:
        if len(curr_elements)==1 and curr_elements[-1]==0: n = 1
        elif len(curr_elements)+1 >= len(bag_schedule): n = len(bag_schedule)-1
        else: n = len(curr_elements)+1
        new_bag = np.vstack([curr_bag[0],bag_schedule[n]])
    else: 
        new_bag = curr_bag.copy()
        new_count = min(len(bag_schedule)-1,sum(curr_elements==act_elem)+1)
        new_bag[1,act_elem]=bag_schedule[new_count,act_elem]
    if maskZ:
        new_bag[0,stop_idx]=1
    return new_bag

def propagate_batch(states: StateData, actions: tools.TensorDict, cutoff: float, infbag: bool = False, envs = None, seed: int=0) -> StateData:
    state_list = [state_from_td(state) for state in states.to_data_list()]
    action_list = actions_from_td(actions)
    if infbag: 
        env_idx_list = np.random.randint(0,len(envs),len(state_list))
        next_bag_list = []
        for s,a,e_i in zip(state_list,action_list,env_idx_list):
            env = envs.envs[e_i]
            env.reset_bag_schedule()
            bag_schedule = env.bag_schedule
            next_bag_list.append(next_bag(env.byiter, s.bag.reshape(2,-1), s.elements, bag_schedule, a.element, env.stop_idx, env.maskZ).flatten())
    else: next_bag_list = [None]*len(state_list)
    next_state_list = [propagate(s, a, infbag, b) for s, a, b in zip(state_list, action_list,next_bag_list)]
    next_states_processed = [geometrize_state(s_next, cutoff=cutoff, num_classes=states.node_attrs.shape[1], floats=infbag) for s_next in next_state_list]
    return collate_fn(next_states_processed)
