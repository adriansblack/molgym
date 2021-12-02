# pylint: disable=redefined-outer-name
import io

import ase.build
import ase.data
import ase.io
import numpy as np
import pytest
import torch_geometric

from molgym import data
from molgym.data import AtomicNumberTable
from molgym.data.graph_tools import generate_topology
from molgym.data.trajectory import (Action, generate_sparse_reward_trajectory, propagate_state, State, state_to_atoms)


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
    graph = generate_topology(ethanol.positions, cutoff_distance=1.6)
    assert len(graph.nodes) == 9
    assert len(graph.edges) == 8
    assert min(len(graph[n]) for n in graph.nodes) == 1
    assert max(len(graph[n]) for n in graph.nodes) == 4


def test_disjoint_graph(ethanol):
    with pytest.raises(AssertionError):
        generate_topology(ethanol.positions, cutoff_distance=1.0)


def test_rollout(ethanol):
    z_table = AtomicNumberTable([0, 1, 6, 8])

    state = data.get_state_from_atoms(ethanol, 0, z_table)
    for sars in generate_sparse_reward_trajectory(ethanol, z_table, final_reward=0.0):
        state = propagate_state(state, sars.action)
    atoms = state_to_atoms(state, z_table)

    assert ethanol.symbols.get_chemical_formula() == atoms.symbols.get_chemical_formula()
    aligned_atoms = rotation_translation_align(atoms, target=ethanol)
    rmsd = compute_rmsd(aligned_atoms, ethanol)
    assert rmsd < 1e-10


def test_trajectory_generation(ethanol):
    z_table = AtomicNumberTable([0, 1, 6, 8])
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
    assert sum(sars_last.next_state.bag) == 1 and sars_last.next_state.bag[0] == 1  # no atoms remaining, only sentinel


def test_propagate():
    state = State(elements=np.array([0], dtype=int), positions=np.zeros((1, 3)), bag=(0, 1, 1))
    action = Action(focus=0, element=1, distance=1.5, orientation=np.array([1.5, 1.0, 1.2]))
    new_state = propagate_state(state, action)
    assert len(new_state.elements) == 1

    action = Action(focus=0, element=0, distance=1.5, orientation=np.array([1.5, 1.0, 1.2]))
    with pytest.raises(ValueError):
        propagate_state(state, action)


def test_conversions(ethanol):
    z_table = AtomicNumberTable([0, 1, 6, 8])
    sars_list = generate_sparse_reward_trajectory(ethanol, z_table, final_reward=0.0)
    batch_size = 5

    dataset = [data.build_state_action_data(state=sars.state, cutoff=1.7, action=sars.action) for sars in sars_list]
    loader = torch_geometric.loader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
    )

    for i, batch in enumerate(loader):
        sars_items = sars_list[i * batch_size:(i + 1) * batch_size]
        actions = data.get_actions_from_td(batch)
        states = [data.get_state_from_td(item) for item in batch.to_data_list()]
        for sars, action, state in zip(sars_items, actions, states):
            # Action
            assert sars.action.focus == action.focus
            assert sars.action.element == action.element
            assert np.isclose(sars.action.distance, action.distance)
            assert np.allclose(sars.action.orientation, action.orientation)

            # State
            assert np.allclose(sars.state.positions, state.positions)
            assert np.allclose(sars.state.elements, state.elements)
            assert np.allclose(sars.state.bag, state.bag)
