import abc
import time
from typing import Tuple, Dict, Optional, Any

import ase.data
import numpy as np

from molgym.data import Symbols, Positions
from .calculator import Sparrow


def get_minimum_spin_multiplicity(symbols: Symbols) -> int:
    return sum(ase.data.atomic_numbers[s] for s in symbols) % 2 + 1


class MolecularReward(abc.ABC):
    @abc.abstractmethod
    def calculate(self, symbols: Symbols, positions: Positions, gradients=False) -> Tuple[float, dict]:
        raise NotImplementedError


class SparseInteractionReward(MolecularReward):
    def __init__(self) -> None:
        # Due to some mysterious bug in Sparrow, calculations get slower and slower over time.
        # For this reason, we generate a new Sparrow object every time.
        self.calculator = Sparrow('PM6')

        self.settings = {
            'molecular_charge': 0,
            'max_scf_iterations': 128,
            'unrestricted_calculation': 1,
        }

        self.atomic_energies: Dict[str, float] = {}

    def calculate(self, symbols: Symbols, positions: Positions, gradients=False) -> Tuple[float, dict]:
        start = time.time()
        self.calculator = Sparrow('PM6')

        e_tot, grad = self._calculate_properties(symbols, positions, gradients=gradients)
        e_parts = sum(self._calculate_atomic_energy(s) for s in symbols)
        e_int = e_tot - e_parts

        reward = -1 * e_int

        info: Dict[str, Any] = {
            'elapsed_time': time.time() - start,
            'reward': reward,
        }

        if grad is not None:
            info['gradients'] = -1 * grad

        return reward, info

    def _calculate_atomic_energy(self, s: str) -> float:
        if s not in self.atomic_energies:
            self.atomic_energies[s], _forces = self._calculate_properties(
                symbols=s,
                positions=np.zeros((1, 3), dtype=float),
                gradients=False,
            )
        return self.atomic_energies[s]

    def _calculate_properties(
        self,
        symbols: Symbols,
        positions: Positions,
        gradients: bool,
    ) -> Tuple[float, Optional[np.ndarray]]:
        if len(symbols) == 0:
            return 0.0, np.zeros((0, 3), dtype=float) if gradients else None

        self.calculator.set_elements(list(symbols))
        self.calculator.set_positions(positions)
        self.settings['spin_multiplicity'] = get_minimum_spin_multiplicity(symbols)
        self.calculator.set_settings(self.settings)
        energy = self.calculator.calculate_energy()

        # energy and forces
        return energy, self.calculator.calculate_gradients() if gradients else None
