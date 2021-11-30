from dataclasses import dataclass
from typing import Sequence, Tuple, List, Optional

import ase.data
import numpy as np

from molgym import tools
from . import tables

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


def propagate_finite_bag_state(state: State, action: Action) -> State:
    bag = tables.remove_element_from_bag(action.element, state.bag)

    if (len(state.elements) == 0) or (len(state.elements) == 1 and state.elements[0] == 0):
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


def get_canvas(atoms: ase.Atoms, z_table: tables.AtomicNumberTable) -> Tuple[np.ndarray, np.ndarray]:
    if len(atoms) > 0:
        return (
            np.array([z_table.z_to_index(ase.data.atomic_numbers[s]) for s in atoms.symbols], dtype=int),
            atoms.positions,
        )

    # Empty canvases need to contain the "dummy" atom
    return np.array([0], dtype=int), np.array([[0.0, 0.0, 0.0]], dtype=float)


def get_empty_canvas_state(atoms: ase.Atoms, z_table: tables.AtomicNumberTable) -> State:
    elements, positions = get_canvas(ase.Atoms(), z_table)
    bag = tables.bag_from_atomic_numbers(zs=(ase.data.atomic_numbers[s] for s in atoms.symbols), z_table=z_table)
    return State(elements, positions, bag)


def get_empty_bag_state(atoms: ase.Atoms, z_table: tables.AtomicNumberTable) -> State:
    elements, positions = get_canvas(atoms, z_table)
    empty_bag = tables.bag_from_atomic_numbers(zs=[], z_table=z_table)
    return State(elements, positions, empty_bag)


def state_to_atoms(state: State, z_table: tables.AtomicNumberTable) -> ase.Atoms:
    return ase.Atoms(
        symbols=[ase.data.chemical_symbols[z_table.index_to_z(e)] for e in state.elements],
        positions=state.positions,
        info={'bag': {ase.data.chemical_symbols[z_table.index_to_z(i)]: v
                      for i, v in enumerate(state.bag)}},
    )


def get_default_action(element: int) -> Action:
    return Action(focus=0, element=element, distance=1.5, orientation=np.array([1.0, 0.0, 0.0]))


def get_last_action(
    elements: np.ndarray,
    positions: np.ndarray,
    focus: Optional[int],
) -> Action:
    assert len(elements) > 0

    if focus is None:
        distance = np.linalg.norm(positions[:-1] - positions[-1], axis=-1, keepdims=False)  # [n_atoms, ]
        focus = np.argmin(distance).item()

    distance = np.linalg.norm(positions[focus] - positions[-1])
    orientation = -positions[focus] + positions[-1]
    orientation /= np.linalg.norm(orientation)

    return Action(
        focus=focus,
        element=elements[-1],
        distance=distance,
        orientation=orientation,
    )


def generate_sparse_reward_trajectory(
    atoms: ase.Atoms,
    z_table: tables.AtomicNumberTable,
    final_reward: float,
    focuses: Optional[List[Optional[int]]] = None,
) -> Trajectory:
    if focuses is None:
        focuses = [None] * len(atoms)

    canvases = [get_canvas(atoms[:i], z_table) for i in range(len(atoms) + 1)]
    bags = [
        tables.bag_from_atomic_numbers(zs=(ase.data.atomic_numbers[s] for s in atoms[i:].symbols), z_table=z_table)
        for i in range(len(atoms) + 1)
    ]
    states = [State(elements, positions, bag) for (elements, positions), bag in zip(canvases, bags)]

    actions = [get_default_action(states[1].elements[0])] + [
        get_last_action(elements=state.elements, positions=state.positions, focus=focus)
        for state, focus in zip(states[2:], focuses[1:])
    ]

    num_actions = len(actions)
    assert num_actions == len(states) - 1

    tau = []
    for index, (state, action, next_state) in enumerate(zip(states[:-1], actions, states[1:])):
        tau.append(
            SARS(
                state=state,
                action=action,
                reward=final_reward if index == num_actions - 1 else 0.0,
                next_state=next_state,
                done=index == num_actions - 1,
            ))

    return tau


def get_actions_from_td(td: tools.TensorDict) -> List[Action]:
    return [
        Action(focus=f, element=e, distance=d, orientation=o) for f, e, d, o in zip(
            tools.to_numpy(td[FOCUS_KEY]),
            tools.to_numpy(td[ELEMENT_KEY]),
            tools.to_numpy(td[DISTANCE_KEY]),
            tools.to_numpy(td[ORIENTATION_KEY]),
        )
    ]
