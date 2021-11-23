from dataclasses import dataclass
from typing import Sequence, Tuple, List, Optional

import ase.data
import ase.io
import numpy as np

from molgym.tools import TensorDict, to_numpy
from . import graph_tools, tables


@dataclass
class Action:
    focus: int
    element: int  # index, not Z
    distance: float
    orientation: Tuple[float, float, float]


@dataclass
class State:
    elements: List[int]  # indices, not Zs
    positions: np.ndarray
    bag: tables.Bag


@dataclass
class DiscreteBagState(State):
    bag: tables.DiscreteBag


@dataclass
class SARS:
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


Trajectory = Sequence[SARS]


def get_focus(canvas: ase.Atoms, atom: ase.Atom) -> int:
    # If canvas is emtpy, 0 is returned
    current_min = np.inf
    min_index = 0

    for i, canvas_atom in enumerate(canvas):
        distance = np.linalg.norm(canvas_atom.position - atom.position)

        if distance < current_min:
            current_min = distance
            min_index = i

    return min_index


def get_distance(
        canvas: ase.Atoms,
        focus: int,
        new_atom: ase.Atom,
        default_distance=1.5,  # Angstrom
) -> float:
    # If canvas is emtpy, <default_distance> is returned
    if len(canvas) == 0:
        return default_distance

    focal_atom = canvas[focus]
    return np.linalg.norm(focal_atom.position - new_atom.position)


def get_orientation(
    canvas: ase.Atoms,
    focus: int,
    new_atom: ase.Atom,
) -> Tuple[float, float, float]:
    if len(canvas) == 0:
        return 1.0, 0.0, 0.0

    focal_atom = canvas[focus]
    vec = -focal_atom.position + new_atom.position
    vec /= np.linalg.norm(vec)
    return vec[0], vec[1], vec[2]


def get_action_sequence(
    atoms: ase.Atoms,
    z_table: tables.AtomicNumberTable,
    focuses: Optional[List[int]] = None,
) -> Sequence[Action]:
    if focuses is None:
        focuses = [get_focus(atoms[:t], atoms[t]) for t in range(len(atoms))]

    elements = [z_table.z_to_index(ase.data.atomic_numbers[atom.symbol]) for atom in atoms]
    distances = [get_distance(canvas=atoms[:t], focus=focuses[t], new_atom=atoms[t]) for t in range(len(atoms))]
    orientations = [get_orientation(canvas=atoms[:t], focus=focuses[t], new_atom=atoms[t]) for t in range(len(atoms))]

    return [
        Action(focus=focus, element=element, distance=distance, orientation=orientation)
        for focus, element, distance, orientation in zip(focuses, elements, distances, orientations)
    ]


def reorder_breadth_first(atoms: ase.Atoms, cutoff_distance=1.6, seed=1) -> ase.Atoms:
    graph = graph_tools.generate_topology(atoms, cutoff_distance=cutoff_distance)
    sequence = graph_tools.breadth_first_rollout(graph, seed=seed)
    return graph_tools.select_atoms(atoms, sequence)


def reorder_random_neighbor(atoms: ase.Atoms, cutoff_distance=1.6, seed=1) -> ase.Atoms:
    graph = graph_tools.generate_topology(atoms, cutoff_distance=cutoff_distance)
    sequence = graph_tools.random_neighbor_rollout(graph, seed=seed)
    return graph_tools.select_atoms(atoms, sequence)


def get_canvas(atoms: ase.Atoms, z_table: tables.AtomicNumberTable) -> Tuple[List[int], np.ndarray]:
    if len(atoms) > 0:
        return (
            [z_table.z_to_index(ase.data.atomic_numbers[s]) for s in atoms.symbols],
            atoms.positions,
        )

    return [0], np.array([[0.0, 0.0, 0.0]])


def get_initial_state(atoms: ase.Atoms, z_table: tables.AtomicNumberTable) -> DiscreteBagState:
    elements, positions = get_canvas(ase.Atoms(), z_table)
    bag = tables.discrete_bag_from_atomic_numbers(zs=(ase.data.atomic_numbers[s] for s in atoms.symbols),
                                                  z_table=z_table)
    return DiscreteBagState(elements, positions, bag)


def propagate_discrete_bag_state(state: DiscreteBagState, action: Action) -> DiscreteBagState:
    new_bag = tables.remove_element_from_bag(action.element, state.bag)

    if (len(state.elements) == 0) or (len(state.elements) == 1 and state.elements[0] == 0):
        return DiscreteBagState(elements=[action.element], positions=np.zeros((1, 3)), bag=new_bag)

    new_position = state.positions[action.focus] + action.distance * np.array(action.orientation)
    return DiscreteBagState(
        elements=state.elements + [action.element],
        positions=np.concatenate([state.positions, np.expand_dims(new_position, 0)]),
        bag=new_bag,
    )


def state_to_atoms(state: State, z_table: tables.AtomicNumberTable) -> ase.Atoms:
    return ase.Atoms(
        symbols=[ase.data.chemical_symbols[z_table.index_to_z(e)] for e in state.elements],
        positions=state.positions,
        info={'bag': {ase.data.chemical_symbols[z_table.index_to_z(i)]: v
                      for i, v in enumerate(state.bag)}},
    )


def generate_sparse_reward_trajectory(
    atoms: ase.Atoms,
    z_table: tables.AtomicNumberTable,
    final_reward: float,
    focuses: Optional[List[int]] = None,
) -> Trajectory:
    canvases = [get_canvas(atoms[:i], z_table) for i in range(len(atoms) + 1)]
    bags = [
        tables.discrete_bag_from_atomic_numbers(zs=(ase.data.atomic_numbers[s] for s in atoms[i:].symbols),
                                                z_table=z_table) for i in range(len(atoms) + 1)
    ]
    states = [State(zs, positions, bag) for (zs, positions), bag in zip(canvases, bags)]
    actions = get_action_sequence(atoms, z_table, focuses=focuses)

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


def build_actions(td: TensorDict) -> List[Action]:
    return [
        Action(focus=f, element=e, distance=d, orientation=o) for (f, e, d, o) in zip(
            to_numpy(td['focus']), to_numpy(td['element']), to_numpy(td['distance']), to_numpy(td['orientation']))
    ]
