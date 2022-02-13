import argparse
from typing import Tuple, Optional

import ase.io
import ase.neighborlist
import networkx as nx
import numpy as np
import scipy.constants
import scipy.optimize
from scine_sparrow import Calculation


def minimize(
    calculator,
    atoms: ase.Atoms,
    charge: int,
    spin_multiplicity: int,
    max_iter=120,
    fixed_indices=None,
    verbose=False,
) -> Tuple[ase.Atoms, bool]:
    atoms = atoms.copy()
    calculator.set_elements(list(atoms.symbols))
    calculator.set_settings({'molecular_charge': charge, 'spin_multiplicity': spin_multiplicity})

    mask = np.ones((len(atoms) * 3, ), dtype=float)
    if fixed_indices is not None:
        for index in fixed_indices:
            mask[index * 3:(index + 1) * 3] = 0

    def function(coords: np.ndarray) -> Tuple[float, np.ndarray]:
        calculator.set_positions(coords.reshape(-1, 3))
        energy = calculator.calculate_energy()
        gradients = calculator.calculate_gradients()
        return energy, gradients.flatten() * mask

    initial_coords = atoms.positions.flatten()

    minimize_result = scipy.optimize.minimize(
        function,
        x0=initial_coords,
        jac=True,
        method='BFGS',
        options={
            'maxiter': max_iter,
            'disp': verbose,
            'norm': np.inf,  # equivalent to taking numpy.amax(numpy.abs(gradient))
            'gtol': 3e-4,  # TolMaxG=3e-4 (ORCA)
        },
    )

    atoms.positions = minimize_result.x.reshape(-1, 3)

    return atoms, minimize_result.success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='path to results file or directory', required=True)
    return parser.parse_args()


def convert_to_ang(atoms: ase.Atoms) -> ase.Atoms:
    copy = atoms.copy()
    angstrom_per_bohr = scipy.constants.value('Bohr radius') / scipy.constants.angstrom
    copy.positions *= angstrom_per_bohr
    return copy


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


def cap_off(atoms: ase.Atoms) -> ase.Atoms:
    graph = generate_topology(atoms.positions, cutoff_distance=2.0)
    assert all(len(graph[n]) <= 3 for n in graph)

    copy = atoms.copy()
    for n in graph:
        neighbors = graph[n]
        if len(neighbors) < 3:
            assert len(neighbors) == 2

            ps = np.zeros(shape=(2, 3), dtype=float)
            for i, neighbor in enumerate(neighbors):
                ps[i] = -atoms.positions[neighbor] + atoms.positions[n]

            p = atoms.positions[n] + np.sum(ps, axis=0)
            copy.append(ase.Atom(symbol='H', position=p))

    return copy


def shift_to_origin(atoms: ase.Atoms) -> ase.Atoms:
    copy = atoms.copy()
    com = np.mean(atoms.positions, axis=0)
    copy.positions -= com
    return copy


def sort_by_distance(atoms: ase.Atoms) -> ase.Atoms:
    # Atoms closest to the center last
    copy = atoms.copy()
    d = np.linalg.norm(copy.positions, ord=2, axis=1)
    indices_sort = np.argsort(-d, axis=-1)
    ordered = copy[indices_sort]
    return ordered


def main():
    calculator = Calculation('PM6')

    args = parse_args()
    s = ase.io.read(args.path)
    s = convert_to_ang(s)
    s = shift_to_origin(s)
    s = sort_by_distance(s)

    # Remove x outer rings
    s = s[6 * (11 + 9 + 7 + 5):]

    s = cap_off(s)
    s, success = minimize(calculator, s, charge=0, spin_multiplicity=1)
    print('Optimized structure successfully')

    s = shift_to_origin(s)
    s = sort_by_distance(s)

    # Remove y inner rings
    k = 4
    s = s[:-k]

    s.info['bag'] = {'C': k}

    # focusable = [False] * (len(s) - 6 * 3) + [True] * (6 * 3)
    focusable = [False] * (len(s) - k) + [True] * k
    assert len(focusable) == len(s)
    s.info['focusable'] = focusable

    ase.io.write('output.xyz', images=s, format='extxyz')


if __name__ == '__main__':
    main()
