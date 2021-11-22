import os

import ase.io
import numpy as np
import pkg_resources
from ase import Atoms

from molgym.rl.calculator import Sparrow


def test_calculator():
    calculator = Sparrow('PM6')
    atoms = Atoms(symbols='HH', positions=[(0, 0, 0), (1.2, 0, 0)])
    calculator.set_elements(list(atoms.symbols))
    calculator.set_positions(atoms.positions)
    calculator.set_settings({'molecular_charge': 0, 'spin_multiplicity': 1})

    gradients = calculator.calculate_gradients()
    energy = calculator.calculate_energy()

    assert np.isclose(energy, -0.9379853016)
    assert gradients.shape == (2, 3)


def test_atomic_energies():
    calculator = Sparrow('PM6')
    calculator.set_positions([(0, 0, 0)])

    calculator.set_elements(['H'])
    calculator.set_settings({'molecular_charge': 0, 'spin_multiplicity': 2})
    assert np.isclose(calculator.calculate_energy(), -0.4133180865)

    calculator.set_elements(['C'])
    calculator.set_settings({'molecular_charge': 0, 'spin_multiplicity': 1})
    assert np.isclose(calculator.calculate_energy(), -4.162353543)

    calculator.set_elements(['O'])
    calculator.set_settings({'molecular_charge': 0, 'spin_multiplicity': 1})
    assert np.isclose(calculator.calculate_energy(), -10.37062419)


def test_energy_gradients():
    calculator = Sparrow('PM6')
    resources_path = pkg_resources.resource_filename(__package__, 'resources')
    atoms = ase.io.read(filename=os.path.join(resources_path, 'h2o.xyz'), format='xyz', index=0)
    calculator.set_positions(atoms.positions)
    calculator.set_elements(list(atoms.symbols))
    calculator.set_settings({'molecular_charge': 0, 'spin_multiplicity': 1})

    energy = calculator.calculate_energy()
    gradients = calculator.calculate_gradients()

    energy_file = os.path.join(resources_path, 'energy.dat')
    expected_energy = float(np.genfromtxt(energy_file))
    assert np.isclose(energy, expected_energy)

    gradients_file = os.path.join(resources_path, 'gradients.dat')
    expected_gradients = np.genfromtxt(gradients_file)
    assert np.allclose(gradients, expected_gradients)
