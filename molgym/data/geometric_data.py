from typing import Optional

import numpy as np
import torch.utils.data
import torch_geometric

from molgym.tools import to_one_hot
from .neighborhood import get_neighborhood
from .tables import AtomicNumberTable
from .trajectory import State, Action
from .utils import config_from_atoms, Configuration


def atomic_numbers_to_index_array(atomic_numbers: np.ndarray, z_table: AtomicNumberTable) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)


class AtomicData(torch_geometric.data.Data):
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor


class AtomicBatch(torch_geometric.data.Batch, AtomicData):
    pass


class EnergyForcesData(AtomicData):
    forces: torch.Tensor
    energy: torch.Tensor


class EnergyForcesBatch(torch_geometric.data.Batch, EnergyForcesData):
    pass


def build_energy_forces_data(
    config: Configuration,
    z_table: AtomicNumberTable,
    cutoff: float,
) -> EnergyForcesData:
    edge_index, shifts = get_neighborhood(positions=config.positions, cutoff=cutoff)

    indices = atomic_numbers_to_index_array(config.atomic_numbers, z_table=z_table)
    one_hot_attrs = to_one_hot(
        indices=torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(z_table),
    )

    return EnergyForcesData(
        num_nodes=len(config.atomic_numbers),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
        node_attrs=one_hot_attrs.to(torch.get_default_dtype()),
        positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
        forces=torch.tensor(config.forces, dtype=torch.get_default_dtype()) if config.forces is not None else None,
        energy=torch.tensor(config.energy, dtype=torch.get_default_dtype()) if config.energy is not None else None,
    )


class StateActionData(AtomicData):
    # Bag
    bag: torch.Tensor

    # Action
    focus: torch.Tensor
    z: torch.Tensor
    distance: torch.Tensor
    orientation: torch.Tensor


class StateActionBatch(torch_geometric.data.Batch, StateActionData):
    pass


def build_state_action_data(
    state: State,
    z_table: AtomicNumberTable,
    cutoff: float,
    action: Optional[Action] = None,
) -> StateActionData:
    config = config_from_atoms(state.atoms)

    edge_index, shifts = get_neighborhood(positions=config.positions, cutoff=cutoff)

    indices = atomic_numbers_to_index_array(config.atomic_numbers, z_table=z_table)
    one_hot_attrs = to_one_hot(
        indices=torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
        num_classes=len(z_table),
    )

    return StateActionData(
        # Canvas
        num_nodes=len(state.atoms),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
        node_attrs=one_hot_attrs.to(torch.get_default_dtype()),
        positions=torch.tensor(config.positions, dtype=torch.get_default_dtype()),
        # Bag
        bag=torch.tensor([state.bag], dtype=torch.long),
        # Action (optional)
        focus=torch.tensor(action.focus, dtype=torch.long) if action else None,
        z=torch.tensor(z_table.z_to_index(action.z), dtype=torch.long) if action else None,
        distance=torch.tensor(action.distance, dtype=torch.get_default_dtype()) if action else None,
        orientation=torch.tensor([action.orientation], dtype=torch.get_default_dtype()) if action else None,
    )
