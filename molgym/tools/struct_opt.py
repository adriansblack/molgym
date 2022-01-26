import dataclasses
from typing import Tuple, List, Optional

import ase.data
import ase.io
import ase.neighborlist
import numpy as np
import scipy.constants
import scipy.optimize
from scine_sparrow import Calculation


@dataclasses.dataclass
class Configuration:
    atomic_numbers: np.ndarray
    positions: np.ndarray
    energy: float
    forces: np.ndarray


def minimize(
    calculator: Calculation,
    atomic_numbers: np.ndarray,
    positions: np.ndarray,
    charge: int,
    spin_multiplicity: int,
    max_iter=120,  # this is not the same as the number of functions calls!
    fixed: Optional[List[bool]] = None,
    verbose=False,
) -> Tuple[List[Configuration], bool]:
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    assert len(atomic_numbers.shape) == 1
    assert positions.shape[0] == atomic_numbers.shape[0]
    assert fixed is None or len(fixed) == positions.shape[0]

    calculator.set_elements([ase.data.chemical_symbols[z] for z in atomic_numbers])
    calculator.set_settings({'molecular_charge': charge, 'spin_multiplicity': spin_multiplicity})

    if fixed is not None:
        fixed_array = np.tile(np.array([fixed], dtype=bool).T, (1, 3))  # [n_atoms, 3]
        mask = (~fixed_array).astype(float).flatten()  # [3 * n_atoms, ]
    else:
        mask = np.array(1.0)

    configs: List[Configuration] = []

    def function(coords: np.ndarray) -> Tuple[float, np.ndarray]:
        current_positions = coords.reshape(-1, 3)
        calculator.set_positions(current_positions)
        energy = calculator.calculate_energy()
        gradients = calculator.calculate_gradients()

        configs.append(
            Configuration(
                atomic_numbers=atomic_numbers,
                positions=current_positions,
                energy=energy,
                forces=gradients,
            ))

        return energy, gradients.flatten() * mask

    minimize_result = scipy.optimize.minimize(
        function,
        x0=positions.flatten(),
        jac=True,
        method='BFGS',
        options={
            'maxiter': max_iter,
            'disp': verbose,
            'norm': np.inf,  # equivalent to taking numpy.amax(numpy.abs(gradient))
            'gtol': 3e-4,  # TolMaxG=3e-4 (ORCA)
        },
    )

    return configs, minimize_result.success
