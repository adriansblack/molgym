import ase.data
import numpy as np
import pytest
from ase import Atoms

from molgym import rl


@pytest.fixture
def atoms():
    return Atoms(symbols='OHH',
                 positions=[
                     (-0.27939703, 0.83823215, 0.00973345),
                     (-0.52040310, 1.77677325, 0.21391146),
                     (0.54473632, 0.90669722, -0.53501306),
                 ])


@pytest.fixture
def charge() -> int:
    return 0


@pytest.fixture
def spin_multiplicity() -> int:
    return 1


def test_minimize(atoms, charge, spin_multiplicity):
    reward_fn = rl.SparseInteractionReward()

    reward1, info1 = reward_fn.calculate(atoms.symbols, atoms.positions, gradients=True)

    configs, success = rl.optimize_structure(reward_fn=reward_fn, symbols=atoms.symbols, positions=atoms.positions)
    assert success

    reward2, info2 = reward_fn.calculate(configs[-1].symbols, configs[-1].positions, gradients=True)

    assert reward2 > reward1
    assert np.sum(np.square(info1['gradients'])) > np.sum(np.square(info2['gradients']))
    assert np.all(info2['gradients'] < 1E-3)


def test_minimize_fail(atoms, charge, spin_multiplicity):
    reward_fn = rl.SparseInteractionReward()

    _configs, success = rl.optimize_structure(
        reward_fn=reward_fn,
        symbols=atoms.symbols,
        positions=atoms.positions,
        max_iter=1,
    )

    assert not success


def test_minimize_fixed(atoms, charge, spin_multiplicity):
    reward_fn = rl.SparseInteractionReward()

    configs, success = rl.optimize_structure(
        reward_fn=reward_fn,
        symbols=atoms.symbols,
        positions=atoms.positions,
        fixed=[False, False, True],
    )

    assert success
    assert np.all((atoms.positions - configs[-1].positions)[-1] < 1E-6)
