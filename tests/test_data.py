import numpy as np
import torch_geometric

from molgym.data import Configuration, get_neighborhood, AtomicNumberTable
from molgym.data.geometric_data import atomic_numbers_to_index_array, geometrize_config


def test_conversion():
    table = AtomicNumberTable(zs=[0, 1, 8])
    array = np.array([8, 8, 1])
    indices = atomic_numbers_to_index_array(array, z_table=table)
    expected = np.array([2, 2, 1], dtype=int)
    assert np.allclose(expected, indices)


class TestConfiguration:
    table = AtomicNumberTable([0, 1, 8])
    config = Configuration(atomic_numbers=np.array([8, 1, 1]),
                           positions=np.array([
                               [0.0, -2.0, 0.0],
                               [1.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0],
                           ]))

    def test_atomic_data(self):
        data = geometrize_config(self.config, z_table=self.table, cutoff=3.0)

        assert data.edge_index.shape == (2, 4)
        assert data.node_attrs.shape == (3, 3)

    def test_collate(self):
        data1 = geometrize_config(self.config, z_table=self.table, cutoff=3.0)
        data2 = geometrize_config(self.config, z_table=self.table, cutoff=3.0)

        assert torch_geometric.loader.DataLoader(dataset=[data1, data2], batch_size=32)


def test_neighborhood_basics():
    positions = np.array([
        [-1.0, 0.0, 0.0],
        [+0.0, 0.0, 0.0],
        [+1.0, 0.0, 0.0],
    ])

    indices, shifts = get_neighborhood(positions, cutoff=1.5)
    assert indices.shape == (2, 4)
    assert shifts.shape == (4, 3)


def test_neighborhood_signs():
    positions = np.array([
        [+0.5, 0.5, 0.0],
        [+1.0, 1.0, 0.0],
    ])

    cell = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    edge_index, shifts = get_neighborhood(positions, cutoff=3.5, pbc=(True, False, False), cell=cell)
    num_edges = 10
    assert edge_index.shape == (2, num_edges)
    assert shifts.shape == (num_edges, 3)
