import ase.data
import numpy as np
import pytest
from ase import Atoms
from molgym import tools

from molgym.rl.calculator import Sparrow


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
    calculator = Sparrow('PM6')

    calculator.set_elements(list(atoms.symbols))
    calculator.set_positions(atoms.positions)
    calculator.set_settings({'molecular_charge': charge, 'spin_multiplicity': spin_multiplicity})
    energy1 = calculator.calculate_energy()
    gradients1 = calculator.calculate_gradients()

    configs, success = tools.minimize(
        calculator=calculator,
        atomic_numbers=np.array([ase.data.atomic_numbers[s] for s in atoms.symbols]),
        positions=atoms.positions,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
    )

    assert success

    calculator.set_positions(configs[-1].positions)
    energy2 = calculator.calculate_energy()
    gradients2 = calculator.calculate_gradients()

    assert energy1 > energy2
    assert np.sum(np.square(gradients1)) > np.sum(np.square(gradients2))
    assert np.all(gradients2 < 1E-3)


def test_minimize_fail(atoms, charge, spin_multiplicity):
    calculator = Sparrow('PM6')
    calculator.set_elements(list(atoms.symbols))
    calculator.set_positions(atoms.positions)
    calculator.set_settings({'molecular_charge': charge, 'spin_multiplicity': spin_multiplicity})

    _configs, success = tools.minimize(
        calculator=calculator,
        atomic_numbers=np.array([ase.data.atomic_numbers[s] for s in atoms.symbols]),
        positions=atoms.positions,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        max_iter=1,
    )

    assert not success


def test_minimize_fixed(atoms, charge, spin_multiplicity):
    calculator = Sparrow('PM6')

    calculator.set_elements(list(atoms.symbols))
    calculator.set_positions(atoms.positions)
    calculator.set_settings({'molecular_charge': charge, 'spin_multiplicity': spin_multiplicity})

    configs, success = tools.minimize(
        calculator=calculator,
        atomic_numbers=np.array([ase.data.atomic_numbers[s] for s in atoms.symbols]),
        positions=atoms.positions,
        charge=charge,
        spin_multiplicity=spin_multiplicity,
        fixed=[False, False, True],
    )

    assert success
    assert np.all((atoms.positions - configs[-1].positions)[-1] < 1E-6)
