import numpy as np

from molgym.data import Configuration, AtomicData, get_data_loader, get_neighborhood
from molgym.data.atomic_data import atomic_numbers_to_index_array
from molgym.data.tables import AtomicNumberTable


def test_conversion():
    table = AtomicNumberTable(zs=[1, 8])
    array = np.array([8, 8, 1])
    indices = atomic_numbers_to_index_array(array, z_table=table)
    expected = np.array([1, 1, 0], dtype=int)
    assert np.allclose(expected, indices)


class TestAtomicData:
    table = AtomicNumberTable([1, 8])
    config = Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array([
            [0.0, -2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]),
        forces=np.array([
            [0.0, -1.3, 0.0],
            [1.0, 0.2, 0.0],
            [0.0, 1.1, 0.3],
        ]),
        energy=-1.5,
    )

    def test_atomic_data(self):
        data = AtomicData.from_config(self.config, z_table=self.table, cutoff=3.0)

        assert data.edge_index.shape == (2, 4)
        assert data.forces.shape == (3, 3)
        assert data.node_attrs.shape == (3, 2)

    def test_collate(self):
        data1 = AtomicData.from_config(self.config, z_table=self.table, cutoff=3.0)
        data2 = AtomicData.from_config(self.config, z_table=self.table, cutoff=3.0)

        data_loader = get_data_loader([data1, data2], batch_size=32)
        assert data_loader


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
