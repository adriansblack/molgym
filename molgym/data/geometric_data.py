from typing import Optional, List, Dict, Union

import numpy as np
import torch.utils.data
import torch_geometric

from molgym import tools
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


def atomic_numbers_to_index_array(atomic_numbers: np.ndarray, z_table: utils.AtomicNumberTable) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)


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


def tensorize_bag(bag: Bag) -> tools.TensorDict:
    return dict(bag=torch.tensor([bag], dtype=torch.long))


def geometrize_config(
    config: utils.Configuration,
    z_table: utils.AtomicNumberTable,
    cutoff: float,
) -> CanvasData:
    element_indices = atomic_numbers_to_index_array(config.atomic_numbers, z_table=z_table)
    info = tensorize_canvas(element_indices, config.positions, cutoff=cutoff, num_classes=len(z_table))
    return CanvasData(**info)


def geometrize_state(state: State, cutoff: float) -> StateData:
    return StateData(
        **tensorize_canvas(state.elements, state.positions, cutoff=cutoff, num_classes=len(state.bag)),
        **tensorize_bag(state.bag),
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
) -> Dict[str, Union[tools.TensorDict, StateData]]:
    info: Dict[str, Union[tools.TensorDict, StateData]] = {'state': geometrize_state(state, cutoff)}

    if action:
        info['action'] = tensorize_action(action)

    return info


def process_sars(
    sars: trajectory.SARS,
    cutoff: float,
) -> Dict[str, Union[torch.Tensor, tools.TensorDict, StateData]]:
    return {
        'state': geometrize_state(sars.state, cutoff=cutoff),
        'action': tensorize_action(sars.action),
        'reward': torch.tensor(sars.reward, dtype=torch.get_default_dtype()),
        'done': torch.tensor(sars.done, dtype=torch.bool),
        'next_state': geometrize_state(sars.next_state, cutoff=cutoff),
    }


def state_from_td(td: tools.TensorDict) -> State:
    return State(
        elements=tools.to_numpy(td[ELEMENTS_KEY]).reshape(-1),
        positions=tools.to_numpy(td[POSITIONS_KEY]).reshape(-1, 3),
        bag=tools.to_numpy(td[BAG_KEY]).reshape(-1),
    )


def propagate_batch(states: StateData, actions: tools.TensorDict, cutoff: float) -> StateData:
    state_list = [state_from_td(state) for state in states.to_data_list()]
    action_list = actions_from_td(actions)
    next_state_list = [propagate(s, a) for s, a in zip(state_list, action_list)]
    next_states_processed = [geometrize_state(s_next, cutoff=cutoff) for s_next in next_state_list]
    return collate_fn(next_states_processed)
