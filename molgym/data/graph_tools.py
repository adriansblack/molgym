import queue
from typing import Sequence

import ase.io
import networkx as nx
import numpy as np

from .neighborhood import get_neighborhood


def generate_topology(
        atoms: ase.Atoms,
        cutoff_distance: float,  # Angstrom
) -> nx.Graph:
    edge_index, _shifts = get_neighborhood(positions=atoms.positions, cutoff=cutoff_distance, pbc=None)
    graph = nx.from_edgelist(edge_index.transpose())

    assert nx.is_connected(graph)
    assert len(graph) == len(atoms)
    return graph


def breadth_first_rollout(graph: nx.Graph, seed: int) -> Sequence[int]:
    rng = np.random.default_rng(seed)
    start_index = int(rng.choice(graph.nodes))  # cast to int as choice() returns numpy.int64
    visited = [start_index]
    node_queue: queue.SimpleQueue[int] = queue.SimpleQueue()
    node_queue.put(start_index)

    while not node_queue.empty():
        node = node_queue.get()
        neighbors = list(graph.neighbors(node))
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.append(neighbor)
                node_queue.put(neighbor)

    assert len(visited) == len(graph)
    return visited


def random_neighbor_rollout(graph: nx.Graph, seed: int) -> Sequence[int]:
    rng = np.random.default_rng(seed)
    start_index = int(rng.choice(graph.nodes))  # cast to int as choice() returns numpy.int64
    visited = [start_index]
    next_nodes = [start_index]

    while len(next_nodes) > 0:
        node = rng.choice(next_nodes)
        next_nodes.remove(node)
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                visited.append(neighbor)
                next_nodes.append(neighbor)

    assert len(visited) == len(graph)
    return visited


def select_atoms(atoms: ase.Atoms, indices: Sequence[int]) -> ase.Atoms:
    new_atoms = ase.Atoms()
    for index in indices:
        new_atoms.append(atoms[index])
    return new_atoms
