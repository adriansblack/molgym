from dataclasses import dataclass
from typing import Iterable, Sequence, List, Optional, Dict, Union

import ase.data
import numpy as np

from .utils import Elements, Positions, AtomicNumberTable

POSITIONS_KEY = 'positions'
ELEMENTS_KEY = 'elements'
BAG_KEY = 'bag'

FOCUS_KEY = 'focus'
ELEMENT_KEY = 'element'
DISTANCE_KEY = 'distance'
ORIENTATION_KEY = 'orientation'

Bag = np.ndarray


def bag_from_atomic_numbers(zs: Iterable[int], z_table: AtomicNumberTable) -> Bag:
    bag = [0] * len(z_table)
    for z in zs:
        bag[z_table.z_to_index(z)] += 1

    return np.array(bag, dtype=int)


def remove_element_from_bag(e: int, bag: Bag) -> Bag:
    if bag[e] < 1:
        raise ValueError(f"Cannot remove element with index '{e}' from '{bag}'")

    copy = bag.copy()
    copy[e] -= 1
    return copy


def add_element_to_bag(e: int, bag: Bag) -> Bag:
    copy = bag.copy()
    copy[e] += 1
    return copy


def bag_is_empty(bag: Bag) -> bool:
    return np.all(bag < 1)  # type: ignore


def no_real_atoms_in_bag(bag: Bag) -> bool:
    return np.all(bag[1:] < 1)  # type: ignore


@dataclass
class Action:
    focus: int
    element: int  # index, not Z
    distance: float
    orientation: np.ndarray


@dataclass
class State:
    elements: Elements  # indices, not Zs
    positions: Positions
    bag: Bag


@dataclass
class SARS:
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


Trajectory = Sequence[SARS]


def propagate(state: State, action: Action) -> State:
    bag = remove_element_from_bag(action.element, state.bag)

    # If bag is empty, add sentinel element at position 0
    if bag_is_empty(bag):
        bag = add_element_to_bag(0, bag)

    if len(state.elements) == 1 and state.elements[0] == 0:
        return State(
            elements=np.array([action.element], dtype=int),
            positions=np.zeros((1, 3), dtype=float),
            bag=bag,
        )

    new_position = state.positions[action.focus] + action.distance * action.orientation
    return State(
        elements=np.concatenate([state.elements, np.array([action.element])]),
        positions=np.concatenate([state.positions, np.expand_dims(new_position, axis=0)], axis=0),
        bag=bag,
    )


def state_to_atoms(state: State, z_table: AtomicNumberTable, info: Optional[Dict] = None) -> ase.Atoms:
    d = {'bag': {ase.data.chemical_symbols[z_table.index_to_z(i)]: int(v) for i, v in enumerate(state.bag)}}
    if info is not None:
        d.update(info)
    return ase.Atoms(
        symbols=[ase.data.chemical_symbols[z_table.index_to_z(e)] for e in state.elements],
        positions=state.positions,
        info=d,
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


def get_state_from_atoms(atoms: ase.Atoms, index: int, z_table: AtomicNumberTable) -> State:
    placed, remaining = atoms[:index], atoms[index:]
    if index == 0:
        elements = np.array([0], dtype=int)
        positions = np.array([[0.0, 0.0, 0.0]], dtype=float)
    else:
        elements = np.array([z_table.z_to_index(ase.data.atomic_numbers[s]) for s in placed.symbols])
        positions = placed.positions

    if index >= len(atoms):
        bag = bag_from_atomic_numbers(zs=[0], z_table=z_table)
    else:
        bag = bag_from_atomic_numbers(zs=(ase.data.atomic_numbers[s] for s in remaining.symbols), z_table=z_table)
    return State(elements, positions, bag)


def generate_sparse_reward_trajectory(
    atoms: ase.Atoms,
    z_table: AtomicNumberTable,
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


def analyze_trajectory(tau: Trajectory) -> Dict[str, Union[int, float]]:
    return {
        'length': len(tau),
        'return': sum(sars.reward for sars in tau),
    }


def analyze_trajectories(taus: List[Trajectory]) -> Dict[str, float]:
    dicts = [analyze_trajectory(tau) for tau in taus]

    if len(dicts) == 0:
        return {}

    keys = dicts[0].keys()
    assert all(d.keys() == keys for d in dicts)
    return {key: np.mean([d[key] for d in dicts]) for key in keys}
