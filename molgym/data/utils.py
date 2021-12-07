import logging
from dataclasses import dataclass
from typing import Optional, List, Iterable, Sequence

import ase
import ase.data
import ase.io
import numpy as np

Vector = np.ndarray  # [3,]
Elements = np.ndarray  # [..., ]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom


Configurations = List[Configuration]


def config_from_atoms(atoms: ase.Atoms, energy_key='energy', forces_key='forces') -> Configuration:
    energy = atoms.info.get(energy_key, None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])
    return Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions, energy=energy, forces=forces)


def load_xyz(path: str, formatting: str = 'extxyz') -> List[ase.Atoms]:
    logging.info(f"Loading configurations from '{path}' (format={formatting})")
    atoms_list = ase.io.read(path, ':', format=formatting)
    logging.info(f'Loaded {len(atoms_list)} configurations')
    return atoms_list


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        assert zs[0] == 0
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f'{self.__class__.__name__}: {tuple(self.zs)}'

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: int) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))
