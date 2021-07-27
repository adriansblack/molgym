import ase.data
import pytest

from molgym.data import AtomicNumberTable
from molgym.data.tables import discrete_bag_from_atomic_numbers, remove_z_from_bag


def test_discrete_bag():
    symbols = 'OCCHHOO'
    z_table = AtomicNumberTable([1, 6, 8])
    bag = discrete_bag_from_atomic_numbers((ase.data.atomic_numbers[s] for s in symbols), z_table)

    assert all(value == expected for value, expected in zip(bag, [2, 2, 3]))
    bag = remove_z_from_bag(1, bag, z_table=z_table)
    assert all(value == expected for value, expected in zip(bag, [1, 2, 3]))
    bag = remove_z_from_bag(1, bag, z_table=z_table)
    assert all(value == expected for value, expected in zip(bag, [0, 2, 3]))

    with pytest.raises(ValueError):
        remove_z_from_bag(1, bag, z_table=z_table)


def test_mismatch():
    z_table = AtomicNumberTable([1, 6, 8])
    with pytest.raises(ValueError):
        discrete_bag_from_atomic_numbers(zs=[1, 6, 8, 9], z_table=z_table)
