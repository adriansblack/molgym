import logging
from dataclasses import dataclass
from typing import Optional, List

import ase
import ase.build
import ase.data
import ase.io
import numpy as np

Vector = np.ndarray  # [3,]
Positions = np.ndarray  # [..., 3]
Forces = np.ndarray  # [..., 3]


@dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom


Configurations = List[Configuration]


def rotation_translation_align(atoms: ase.Atoms, target: ase.Atoms) -> ase.Atoms:
    aligned = atoms.copy()
    ase.build.minimize_rotation_and_translation(target, aligned)
    return aligned


def compute_rmsd(a: ase.Atoms, b: ase.Atoms) -> float:
    return np.sqrt(np.mean(np.square(a.positions - b.positions)))


def config_from_atoms(atoms: ase.Atoms, energy_key='energy', forces_key='forces') -> Configuration:
    energy = atoms.info.get(energy_key, None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])
    return Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions, energy=energy, forces=forces)


def load_xyz(path: str, formatting: str = 'extxyz') -> Configurations:
    logging.info(f"Loading dataset from '{path}' (format={formatting})")
    atoms_list = ase.io.read(path, ':', format=formatting)
    configs = [config_from_atoms(atoms) for atoms in atoms_list]
    logging.info(f'Number of configurations: {len(configs)}')
    return configs
