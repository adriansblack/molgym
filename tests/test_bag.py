import ase.data
import numpy as np
import pytest

from molgym.data import SymbolTable
from molgym.data.trajectory import bag_from_symbols, remove_atom_from_bag, bag_from_symbol_count_dict


def test_discrete_bag():
    symbols = 'OCCHHOO'
    s_table = SymbolTable('XHCO')
    bag = bag_from_symbols(symbols, s_table)

    assert all(value == expected for value, expected in zip(bag, [0, 2, 2, 3]))
    bag = remove_atom_from_bag(1, bag)
    assert all(value == expected for value, expected in zip(bag, [0, 1, 2, 3]))
    bag = remove_atom_from_bag(1, bag)
    assert all(value == expected for value, expected in zip(bag, [0, 0, 2, 3]))

    with pytest.raises(ValueError):
        remove_atom_from_bag(1, bag)


def test_mismatch():
    s_table = SymbolTable('XHCO')
    with pytest.raises(ValueError):
        bag_from_symbols(symbols=['H', 'C', 'O', 'F'], s_table=s_table)


def test_bag_from_atomic_numbers_dict():
    s_table = SymbolTable('XHC')
    bag = bag_from_symbol_count_dict({'H': 3, 'C': 2}, s_table)
    assert np.all(bag == np.array([0, 3, 2], dtype=int))

    with pytest.raises(ValueError):
        bag_from_symbol_count_dict({'H': 2, 'O': 1}, s_table)
