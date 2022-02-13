import logging
from dataclasses import dataclass
from typing import Optional, List, Iterable, Sequence

import ase
import ase.data
import ase.io
import numpy as np

Vector = np.ndarray  # [3,] (float)
Symbols = Sequence[str]  # [..., ] (str)
Labels = np.ndarray  # [..., ] (int)
Positions = np.ndarray  # [..., 3] (float)
Forces = np.ndarray  # [..., 3] (float)


@dataclass
class Configuration:
    symbols: Symbols
    positions: Positions  # Angstrom
    energy: Optional[float] = None  # eV
    forces: Optional[Forces] = None  # eV/Angstrom


Configurations = List[Configuration]


def config_from_atoms(atoms: ase.Atoms, energy_key='energy', forces_key='forces') -> Configuration:
    energy = atoms.info.get(energy_key, None)  # eV
    forces = atoms.arrays.get(forces_key, None)  # eV / Ang
    return Configuration(symbols=atoms.symbols, positions=atoms.positions, energy=energy, forces=forces)


def load_xyz(path: str, formatting: str = 'extxyz') -> List[ase.Atoms]:
    logging.info(f"Loading configurations from '{path}' (format={formatting})")
    atoms_list = ase.io.read(path, ':', format=formatting)
    logging.info(f'Loaded {len(atoms_list)} configurations')
    return atoms_list


class SymbolTable:
    def __init__(self, symbols: Symbols) -> None:
        assert symbols[0] == 'X'
        self.symbols = symbols

    def __len__(self) -> int:
        return len(self.symbols)

    def __str__(self) -> str:
        return f'{self.__class__.__name__}: {tuple(self.symbols)}'

    def element_to_symbol(self, e: int) -> str:
        return self.symbols[e]

    def symbol_to_element(self, s: str) -> int:
        return self.symbols.index(s)


def get_atomic_number_table_from_symbols(zs: Iterable[str]) -> SymbolTable:
    symbol_set = set()
    for z in zs:
        symbol_set.add(z)
    return SymbolTable(sorted(list(symbol_set), key=lambda s: ase.data.atomic_numbers[s]))
