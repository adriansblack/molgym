from dataclasses import dataclass
from typing import Sequence, List, Optional

import ase.data
import numpy as np

from . import tables

POSITIONS_KEY = 'positions'
ELEMENTS_KEY = 'elements'
BAG_KEY = 'bag'

FOCUS_KEY = 'focus'
ELEMENT_KEY = 'element'
DISTANCE_KEY = 'distance'
ORIENTATION_KEY = 'orientation'


@dataclass
class Action:
    focus: int
    element: int  # index, not Z
    distance: float
    orientation: np.ndarray


@dataclass
class State:
    elements: np.ndarray  # indices, not Zs
    positions: np.ndarray
    bag: tables.Bag


@dataclass
class SARS:
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


Trajectory = Sequence[SARS]


def propagate_state(state: State, action: Action) -> State:
    bag = tables.remove_element_from_bag(action.element, state.bag)

    # If bag is empty, add sentinel element at position 0
    if tables.bag_is_empty(bag):
        bag = tables.add_element_to_bag(0, bag)

    if len(state.elements) == 1 and state.elements[0] == 0:
        return State(
            elements=np.array([action.element], dtype=int),
            positions=np.zeros((1, 3), dtype=float),
            bag=bag,
        )

    new_position = state.positions[action.focus] + action.distance * np.array(action.orientation)
    return State(
        elements=np.concatenate([state.elements, np.array([action.element])]),
        positions=np.concatenate([state.positions, np.expand_dims(new_position, axis=0)], axis=0),
        bag=bag,
    )


def state_to_atoms(state: State, z_table: tables.AtomicNumberTable) -> ase.Atoms:
    return ase.Atoms(
        symbols=[ase.data.chemical_symbols[z_table.index_to_z(e)] for e in state.elements],
        positions=state.positions,
        info={'bag': {ase.data.chemical_symbols[z_table.index_to_z(i)]: v
                      for i, v in enumerate(state.bag)}},
    )


def get_action(
    state: State,
    element: int,
    position: np.ndarray,
    focus: Optional[int],
) -> Action:
    assert len(state.elements) > 0

    # Replace dummy atom
    if len(state.elements) == 1 and state.elements[0] == 0:
        return Action(focus=0, element=element, distance=1.5, orientation=np.array([1.0, 0.0, 0.0]))

    if focus is None:
        distance = np.linalg.norm(state.positions - position, axis=-1, keepdims=False)  # [n_atoms, ]
        focus = np.argmin(distance).item()

    distance = np.linalg.norm(state.positions[focus] - position)
    orientation = -state.positions[focus] + position
    orientation /= np.linalg.norm(orientation)

    return Action(
        focus=focus,
        element=element,
        distance=distance,
        orientation=orientation,
    )


def get_state_from_atoms(atoms: ase.Atoms, index: int, z_table: tables.AtomicNumberTable) -> State:
    placed, remaining = atoms[:index], atoms[index:]
    if index == 0:
        elements = np.array([0], dtype=int)
        positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    else:
        elements = np.array([z_table.z_to_index(ase.data.atomic_numbers[s]) for s in placed.symbols])
        positions = placed.positions

    if index >= len(atoms):
        bag = tables.bag_from_atomic_numbers(zs=[0], z_table=z_table)
    else:
        bag = tables.bag_from_atomic_numbers(zs=(ase.data.atomic_numbers[s] for s in remaining.symbols),
                                             z_table=z_table)
    return State(elements, positions, bag)


def generate_sparse_reward_trajectory(
    atoms: ase.Atoms,
    z_table: tables.AtomicNumberTable,
    final_reward: float,
    focuses: Optional[List[Optional[int]]] = None,
) -> Trajectory:
    if focuses is None:
        focuses = [None] * len(atoms)

    tau = []
    state = get_state_from_atoms(atoms, index=0, z_table=z_table)
    for i, focus in enumerate(focuses):
        action = get_action(
            state=state,
            focus=focus,
            element=z_table.z_to_index(ase.data.atomic_numbers[atoms[i].symbol]),
            position=atoms[i].position,
        )
        next_state = get_state_from_atoms(atoms, index=i + 1, z_table=z_table)
        tau.append(
            SARS(
                state=state,
                action=action,
                reward=final_reward if i == len(atoms) - 1 else 0.0,
                next_state=next_state,
                done=i == len(atoms) - 1,
            ))

        state = next_state

    return tau
