import ase.data
import numpy as np
import pytest

from molgym.data import AtomicNumberTable
from molgym.data.trajectory import bag_from_atomic_numbers, remove_element_from_bag, bag_from_atomic_numbers_dict


def test_discrete_bag():
    symbols = 'OCCHHOO'
    z_table = AtomicNumberTable([0, 1, 6, 8])
    bag = bag_from_atomic_numbers((ase.data.atomic_numbers[s] for s in symbols), z_table)

    assert all(value == expected for value, expected in zip(bag, [0, 2, 2, 3]))
    bag = remove_element_from_bag(1, bag)
    assert all(value == expected for value, expected in zip(bag, [0, 1, 2, 3]))
    bag = remove_element_from_bag(1, bag)
    assert all(value == expected for value, expected in zip(bag, [0, 0, 2, 3]))

    with pytest.raises(ValueError):
        remove_element_from_bag(1, bag)


def test_mismatch():
    z_table = AtomicNumberTable([0, 1, 6, 8])
    with pytest.raises(ValueError):
        bag_from_atomic_numbers(zs=[1, 6, 8, 9], z_table=z_table)


def test_bag_from_atomic_numbers_dict():
    z_table = AtomicNumberTable([0, 1, 6])
    bag = bag_from_atomic_numbers_dict({1: 3, 6: 2}, z_table)
    assert np.all(bag == np.array([0, 3, 2], dtype=int))

    with pytest.raises(ValueError):
        bag_from_atomic_numbers_dict({1: 2, 8: 1}, z_table)
