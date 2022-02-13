import numpy as np
from ase import Atoms

from molgym import rl


def test_calculation():
    reward_fn = rl.SparseInteractionReward()
    atoms = Atoms('H')
    reward, info = reward_fn.calculate(atoms.symbols, atoms.positions, gradients=False)

    assert np.isclose(reward, 0.0)
    assert 'gradients' not in info


def test_h2():
    atoms = Atoms('HH', positions=[(0, 0, 0), (1, 0, 0)])
    reward_fn = rl.SparseInteractionReward()
    reward, info = reward_fn.calculate(atoms.symbols, atoms.positions, gradients=True)

    assert np.isclose(reward, 0.1696435)
    assert info['gradients'] is not None
