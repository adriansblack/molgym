import itertools
import queue
from typing import Sequence

import ase.io
import networkx as nx
import numpy as np


def generate_topology(
    atoms: ase.Atoms,
    cutoff_distance: float,  # Angstrom
    node_data=False,
) -> nx.Graph:
    graph = nx.Graph()

    if node_data:
        graph.add_nodes_from((i, {'symbol': atom.symbol, 'position': atom.position}) for i, atom in enumerate(atoms))

    for (i, a_i), (j, a_j) in itertools.combinations(list(enumerate(atoms)), 2):
        if np.linalg.norm(a_i.position - a_j.position) < cutoff_distance:
            graph.add_edge(i, j)

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
