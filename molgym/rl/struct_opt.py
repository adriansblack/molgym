import dataclasses
from typing import Tuple, List, Optional

import numpy as np
import scipy.constants
import scipy.optimize

from . import reward


@dataclasses.dataclass
class Configuration:
    zs: np.ndarray
    positions: np.ndarray
    reward: float
    gradients: np.ndarray


def optimize_structure(
    reward_fn: reward.MolecularReward,
    zs: np.ndarray,
    positions: np.ndarray,
    max_iter=120,  # this is not the same as the number of functions calls!
    fixed: Optional[List[bool]] = None,
    verbose=False,
) -> Tuple[List[Configuration], bool]:
    assert len(positions.shape) == 2
    assert positions.shape[1] == 3
    assert len(zs.shape) == 1
    assert positions.shape[0] == zs.shape[0]
    assert fixed is None or len(fixed) == positions.shape[0]

    if fixed is not None:
        fixed_array = np.tile(np.array([fixed], dtype=bool).T, (1, 3))  # [n_atoms, 3]
        mask = (~fixed_array).astype(float).flatten()  # [3 * n_atoms, ]
    else:
        mask = np.array(1.0)

    configs: List[Configuration] = []

    def function(coords: np.ndarray) -> Tuple[float, np.ndarray]:
        current_positions = coords.reshape(-1, 3)
        r, info = reward_fn.calculate(zs, current_positions, gradients=True)
        gradients = info['gradients']

        configs.append(Configuration(
            zs=zs,
            positions=current_positions,
            reward=r,
            gradients=gradients,
        ))

        # We want to maximize the reward (minimize the energy)
        return -1 * r, -1 * gradients.flatten() * mask

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
