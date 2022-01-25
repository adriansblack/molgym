import copy
import queue
from typing import Sequence
from typing import Tuple, Optional

import ase.neighborlist
import networkx as nx
import numpy as np


def get_neighborhood(
    positions: np.ndarray,  # [n, 3]
    cutoff: float,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None:
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, bool) for i in pbc)
    assert cell.shape == (3, 3)

    sender, receiver, unit_shifts = ase.neighborlist.primitive_neighbor_list(
        quantities='ijS',
        pbc=pbc,
        cell=cell,
        positions=positions,
        cutoff=cutoff,
        self_interaction=True,  # we want edges from atom to itself in different periodic images
        use_scaled_positions=False,  # positions are not scaled positions
    )

    if not true_self_interaction:
        # Eliminate self-edges that don't cross periodic boundaries
        true_self_edge = sender == receiver
        true_self_edge &= np.all(unit_shifts == 0, axis=1)
        keep_edge = ~true_self_edge

        # Note: after eliminating self-edges, it can be that no edges remain in this system
        sender = sender[keep_edge]
        receiver = receiver[keep_edge]
        unit_shifts = unit_shifts[keep_edge]

    # Build output
    edge_index = np.stack((sender, receiver))  # [2, n_edges]

    # From the docs: With the shift vector S, the distances D between atoms can be computed from
    # D = positions[j]-positions[i]+S.dot(cell)
    shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]

    return edge_index, shifts


def generate_topology(
        positions: np.ndarray,  # [n_atoms, 3]
        cutoff_distance: float,  # Angstrom
) -> nx.Graph:
    edge_index, _shifts = get_neighborhood(positions=positions, cutoff=cutoff_distance, pbc=None)
    graph = nx.from_edgelist(edge_index.transpose())

    assert nx.is_connected(graph)
    assert len(graph) == positions.shape[0]
    return graph


def breadth_first_rollout(graph: nx.Graph, seed: int, visited: Optional[Sequence[int]] = None) -> Sequence[int]:
    rng = np.random.default_rng(seed)
    node_queue: queue.SimpleQueue[int] = queue.SimpleQueue()

    if visited is None:
        start_index = int(rng.choice(graph.nodes))  # cast to int as choice() returns numpy.int64
        visited = [start_index]
        node_queue.put(start_index)
    else:
        visited = copy.copy(list(visited))
        rng.shuffle(visited)
        for node in visited:
            node_queue.put(node)

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