import numpy as np
import torch_geometric

from molgym.data import Configuration, get_neighborhood, SymbolTable
from molgym.data.geometric_data import geometrize_config


class TestConfiguration:
    table = SymbolTable('XHO')
    config = Configuration(symbols='OHH', positions=np.array([
        [0.0, -2.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]))

    def test_atomic_data(self):
        data = geometrize_config(self.config, s_table=self.table, cutoff=3.0)

        assert data.edge_index.shape == (2, 4)
        assert data.node_attrs.shape == (3, 3)

    def test_collate(self):
        data1 = geometrize_config(self.config, s_table=self.table, cutoff=3.0)
        data2 = geometrize_config(self.config, s_table=self.table, cutoff=3.0)

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
