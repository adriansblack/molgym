import numpy as np

from molgym.data import Configuration, AtomicData, get_data_loader, get_neighborhood

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

table = AtomicNumberTable([1, 8])


class TestAtomicData:
    def test_atomic_data(self):
        data = AtomicData.from_config(config, z_table=table, cutoff=3.0)

        assert data.edge_index.shape == (2, 4)
        assert data.forces.shape == (3, 3)
        assert data.node_attrs.shape == (3, 2)

    def test_collate(self):
        data1 = AtomicData.from_config(config, z_table=table, cutoff=3.0)
        data2 = AtomicData.from_config(config, z_table=table, cutoff=3.0)

        data_loader = get_data_loader([data1, data2], batch_size=32)

        for i in data_loader:
            print(i)


class TestNeighborhood:
    def test_basic(self):
        positions = np.array([
            [-1.0, 0.0, 0.0],
            [+0.0, 0.0, 0.0],
            [+1.0, 0.0, 0.0],
        ])

        indices, shifts = get_neighborhood(positions, cutoff=1.5)
        assert indices.shape == (2, 4)
        assert shifts.shape == (4, 3)

    def test_signs(self):
        positions = np.array([
            [+0.5, 0.5, 0.0],
            [+1.0, 1.0, 0.0],
        ])

        cell = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        edge_index, shifts = get_neighborhood(positions, cutoff=3.5, pbc=(True, False, False), cell=cell)
        num_edges = 10
        assert edge_index.shape == (2, num_edges)
        assert shifts.shape == (num_edges, 3)
