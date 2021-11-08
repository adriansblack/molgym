# pylint: disable=redefined-outer-name
import io
from typing import Sequence

import ase.build
import ase.data
import ase.io
import numpy as np
import pytest

from molgym.data import AtomicNumberTable
from molgym.data.graph_tools import generate_topology
from molgym.data.trajectory import (Action, get_action_sequence, reorder_breadth_first,
                                    generate_sparse_reward_trajectory,
                                    DiscreteBagState, propagate_discrete_bag_state)


def rotation_translation_align(atoms: ase.Atoms, target: ase.Atoms) -> ase.Atoms:
    aligned = atoms.copy()
    ase.build.minimize_rotation_and_translation(target, aligned)
    return aligned


def compute_rmsd(a: ase.Atoms, b: ase.Atoms) -> float:
    return np.sqrt(np.mean(np.square(a.positions - b.positions)))


@pytest.fixture
def ethanol():
    ethanol_string = """9
ethanol
O -1.207454  0.251814  0.015195
C -0.030747 -0.571012  0.019581
C  1.237302  0.268221  0.009126
H -0.076875 -1.243951  0.894697
H -0.152197 -1.158922 -0.910633
H  2.133158 -0.361916 -0.023873
H  1.265320  0.929390 -0.867509
H  1.316863  0.906659  0.895476
H -1.196271  0.874218  0.765639
"""
    return ase.io.read(io.StringIO(ethanol_string), index=0, format='xyz')


@pytest.fixture
def methane():
    methane_string = """5
methane
C   -0.7079    0.0000    0.0000
H    0.7079    0.0000    0.0000
H   -1.0732   -0.7690    0.6852
H   -1.0731   -0.1947   -1.0113
H   -1.0632    0.9786    0.3312
"""
    return ase.io.read(io.StringIO(methane_string), index=0, format='xyz')


def test_graph(ethanol):
    graph = generate_topology(ethanol, cutoff_distance=1.6)
    assert len(graph.nodes) == 9
    assert len(graph.edges) == 8
    assert min(len(graph[n]) for n in graph.nodes) == 1
    assert max(len(graph[n]) for n in graph.nodes) == 4


def test_disjoint_graph(ethanol):
    with pytest.raises(AssertionError):
        generate_topology(ethanol, cutoff_distance=1.0)


def rollout_actions(actions: Sequence[Action], z_table: AtomicNumberTable) -> ase.Atoms:
    atoms = ase.Atoms()

    for action in actions:
        if len(atoms) == 0:
            position = np.array([0.0, 0.0, 0.0])
        else:
            position = atoms[action.focus].position + action.distance * np.array(action.orientation)

        atoms.append(ase.Atom(symbol=z_table.index_to_z(action.element), position=position))

    return atoms


def test_rollout(ethanol):
    atoms = reorder_breadth_first(ethanol, cutoff_distance=1.6, seed=1)
    z_table = AtomicNumberTable([1, 6, 8])
    actions = get_action_sequence(atoms, z_table=z_table)
    rolled_out = rollout_actions(actions, z_table=z_table)
    assert atoms.symbols.get_chemical_formula() == rolled_out.symbols.get_chemical_formula()
    aligned_atoms = rotation_translation_align(rolled_out, target=atoms)
    rmsd = compute_rmsd(aligned_atoms, atoms)
    assert rmsd < 1e-10


def test_trajectory_generation(ethanol):
    z_table = AtomicNumberTable([1, 6, 8])
    trajectory = generate_sparse_reward_trajectory(atoms=ethanol, z_table=z_table, final_reward=1.5)

    assert all(not sars.done for sars in trajectory[:-1])
    assert trajectory[-1].done

    assert all(np.isclose(sars.reward, 0.) for sars in trajectory[:-1])
    assert np.isclose(trajectory[-1].reward, 1.5)

    sars_first = trajectory[0]
    assert len(sars_first.state.elements) == 1  # if canvas is empty, it contains a fake atom
    assert sum(sars_first.state.bag) == len(ethanol)  # bag is full

    sars_last = trajectory[-1]
    assert len(sars_last.state.elements) == len(ethanol) - 1  # canvas close to full
    assert sum(sars_last.state.bag) == 1  # one atom remaining
    assert len(sars_last.next_state.elements) == len(ethanol)  # canvas is full
    assert sum(sars_last.next_state.bag) == 0  # no atoms remaining


def test_propagate():
    state = DiscreteBagState(elements=[0], positions=np.zeros((1, 3)), bag=(0, 1, 1))
    action = Action(focus=0, element=1, distance=1.5, orientation=(1.5, 1.0, 1.2))
    new_state = propagate_discrete_bag_state(state, action)
    assert len(new_state.elements) == 1

    action = Action(focus=0, element=0, distance=1.5, orientation=(1.5, 1.0, 1.2))
    with pytest.raises(ValueError):
        propagate_discrete_bag_state(state, action)
