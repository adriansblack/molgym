from dataclasses import dataclass
from typing import Sequence, Tuple, List

import ase.data
import ase.io
import numpy as np

from . import graph_tools, tables
from .tables import DiscreteBag, Bag, AtomicNumberTable


@dataclass
class Action:
    focus: int
    z: int
    distance: float
    orientation: Tuple[float, float, float]


@dataclass
class State:
    atoms: ase.Atoms
    bag: Bag


@dataclass
class DiscreteBagState:
    atoms: ase.Atoms
    bag: DiscreteBag


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


def get_actions(atoms: ase.Atoms) -> Sequence[Action]:
    focuses = [get_focus(atoms[:t], atoms[t]) for t in range(len(atoms))]
    zs = [ase.data.atomic_numbers[atom.symbol] for atom in atoms]
    distances = [get_distance(canvas=atoms[:t], focus=focuses[t], new_atom=atoms[t]) for t in range(len(atoms))]
    orientations = [get_orientation(canvas=atoms[:t], focus=focuses[t], new_atom=atoms[t]) for t in range(len(atoms))]

    return [
        Action(focus=focus, z=z, distance=distance, orientation=orientation)
        for focus, z, distance, orientation in zip(focuses, zs, distances, orientations)
    ]


def reorder_breadth_first(atoms: ase.Atoms, cutoff_distance=1.6, seed=1) -> ase.Atoms:
    graph = graph_tools.generate_topology(atoms, cutoff_distance=cutoff_distance)
    sequence = graph_tools.breadth_first_rollout(graph, seed=seed)
    return graph_tools.select_atoms(atoms, sequence)


def reorder_random_neighbor(atoms: ase.Atoms, cutoff_distance=1.6, seed=1) -> ase.Atoms:
    graph = graph_tools.generate_topology(atoms, cutoff_distance=cutoff_distance)
    sequence = graph_tools.random_neighbor_rollout(graph, seed=seed)
    return graph_tools.select_atoms(atoms, sequence)


def get_canvases(atoms: ase.Atoms) -> List[ase.Atoms]:
    return [atoms[:i] for i in range(len(atoms) + 1)]


def get_discrete_bags(atoms: ase.Atoms, z_table: AtomicNumberTable) -> List[DiscreteBag]:
    return [
        tables.discrete_bag_from_atomic_numbers(zs=(ase.data.atomic_numbers[s] for s in atoms[i:].symbols),
                                                z_table=z_table) for i in range(len(atoms) + 1)
    ]


def propagate_discrete_bag_state(state: DiscreteBagState, action: Action,
                                 z_table: AtomicNumberTable) -> DiscreteBagState:
    if len(state.atoms) == 0:
        new_position = np.array([0., 0., 0.])
    else:
        new_position = state.atoms[action.focus].position + action.distance * np.array(action.orientation)

    return DiscreteBagState(
        atoms=state.atoms.copy() + ase.Atom(symbol=action.z, position=new_position),
        bag=tables.remove_z_from_bag(action.z, state.bag, z_table),
    )


def generate_sparse_reward_trajectory(atoms: ase.Atoms, z_table: AtomicNumberTable, final_reward: float) -> Trajectory:
    atoms_list = get_canvases(atoms)
    bags = get_discrete_bags(atoms, z_table)

    # Add dummy atom if canvas is empty
    states = [
        State(atoms=atoms if len(atoms) != 0 else ase.Atoms(symbols='X', positions=[[0.0, 0.0, 0.0]]), bag=bag)
        for atoms, bag in zip(atoms_list, bags)
    ]

    actions = get_actions(atoms)
    length = len(actions)

    assert len(actions) == len(states) - 1

    tau = []
    for index, (state, action, next_state) in enumerate(zip(states[:-1], actions, states[1:])):
        tau.append(
            SARS(
                state=state,
                action=action,
                reward=final_reward if index == length - 1 else 0.0,
                next_state=next_state,
                done=index == length - 1,
            ))

    return tau
