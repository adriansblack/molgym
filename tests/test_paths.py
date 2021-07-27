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
from molgym.data.paths import (Action, get_actions, reorder_breadth_first, generate_discrete_bag_construction_path,
                               DiscreteBagState, propagate_state, DiscreteBagStateActionPair)
from molgym.data.utils import rotation_translation_align, compute_rmsd


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


def test_graph_data(ethanol):
    graph = generate_topology(ethanol, cutoff_distance=1.6, node_data=True)
    for atom, (_, node_data) in zip(ethanol, graph.nodes(data=True)):
        assert atom.z == node_data['z']
        assert np.allclose(atom.position, node_data['position'])


def rollout_actions(actions: Sequence[Action]) -> ase.Atoms:
    atoms = ase.Atoms()

    for action in actions:
        if len(atoms) == 0:
            position = np.array([0.0, 0.0, 0.0])
        else:
            position = atoms[action.focus].position + action.distance * np.array(action.orientation)

        atoms.append(ase.Atom(symbol=action.z, position=position))

    return atoms


def test_rollout(ethanol):
    new_atoms = reorder_breadth_first(ethanol, cutoff_distance=1.6, seed=1)
    actions = get_actions(new_atoms)
    rolled_out = rollout_actions(actions)
    aligned_atoms = rotation_translation_align(rolled_out, target=new_atoms)
    rmsd = compute_rmsd(aligned_atoms, new_atoms)
    assert rmsd < 1e-10


def test_path_generation(ethanol):
    z_table = AtomicNumberTable([1, 6, 8])
    path = generate_discrete_bag_construction_path(atoms=ethanol, z_table=z_table)

    p_0 = path[0]
    assert len(p_0.state.atoms) == 0  # canvas is empty
    assert sum(p_0.state.bag) == len(ethanol)  # bag is full

    p_T = path[-1]
    assert len(p_T.state.atoms) == len(ethanol) - 1  # canvas close to full
    assert sum(p_T.state.bag) == 1  # one atom remaining


def test_propagate():
    z_table = AtomicNumberTable([1])
    state = DiscreteBagState(atoms=ase.Atoms(), bag=(1, 1, 1))
    action = Action(focus=0, z=1, distance=1.5, orientation=(1.5, 1.0, 1.2))
    new_state = propagate_state(DiscreteBagStateActionPair(state, action), z_table)
    assert len(new_state.atoms) == 1

    action = Action(focus=0, z=2, distance=1.5, orientation=(1.5, 1.0, 1.2))
    with pytest.raises(ValueError):
        propagate_state(DiscreteBagStateActionPair(state, action), z_table)
