import abc
import time
from typing import Tuple, Dict, Iterable, Optional, Any

import ase.data
import numpy as np

from .calculator import Sparrow


def get_minimum_spin_multiplicity(zs: Iterable[int]) -> int:
    return sum(zs) % 2 + 1


class MolecularReward(abc.ABC):
    @abc.abstractmethod
    def calculate(self, zs: np.ndarray, positions: np.ndarray, gradients=False) -> Tuple[float, dict]:
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

        self.atomic_energies: Dict[int, float] = {}

    def calculate(self, zs: np.ndarray, positions: np.ndarray, gradients=False) -> Tuple[float, dict]:
        start = time.time()
        self.calculator = Sparrow('PM6')

        e_tot, grad = self._calculate_properties(zs, positions, gradients=gradients)
        e_parts = sum(self._calculate_atomic_energy(z) for z in zs)
        e_int = e_tot - e_parts

        reward = -1 * e_int

        info: Dict[str, Any] = {
            'elapsed_time': time.time() - start,
            'reward': reward,
        }

        if grad is not None:
            info['gradients'] = -1 * grad

        return reward, info

    def _calculate_atomic_energy(self, z: int) -> float:
        if z not in self.atomic_energies:
            self.atomic_energies[z], _forces = self._calculate_properties(
                zs=np.array([z], dtype=int),
                positions=np.zeros((1, 3), dtype=float),
                gradients=False,
            )
        return self.atomic_energies[z]

    def _calculate_properties(
        self,
        zs: np.ndarray,
        positions: np.ndarray,
        gradients: bool,
    ) -> Tuple[float, Optional[np.ndarray]]:
        if len(zs) == 0:
            return 0.0, np.zeros((0, 3), dtype=float) if gradients else None

        self.calculator.set_elements(list(ase.data.chemical_symbols[z] for z in zs))
        self.calculator.set_positions(positions)
        self.settings['spin_multiplicity'] = get_minimum_spin_multiplicity(zs)
        self.calculator.set_settings(self.settings)
        energy = self.calculator.calculate_energy()

        # energy and forces
        return energy, self.calculator.calculate_gradients() if gradients else None
