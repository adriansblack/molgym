import io
from unittest import TestCase

import ase.build
import ase.data
import ase.io
import numpy as np

from molgym.data.graph_tools import generate_topology

ethanol_string = """9
ethanol
O -1.207454  0.251814  0.015195
C -0.030747 -0.571012  0.019581
C  1.237302  0.268221  0.009126
H -0.076875 -1.243951  0.894697
H -0.152197 -1.158922 -0.910633
H  2.133158 -0.361916 -0.023873
H  1.265320  0.929390 -0.867509
H  1.316863  0.906659  0.895476
H -1.196271  0.874218  0.765639
"""

methane_string = """5
methane
C   -0.7079    0.0000    0.0000
H    0.7079    0.0000    0.0000
H   -1.0732   -0.7690    0.6852
H   -1.0731   -0.1947   -1.0113
H   -1.0632    0.9786    0.3312
"""


class TopologyTest(TestCase):
    def setUp(self) -> None:
        self.atoms = ase.io.read(io.StringIO(ethanol_string), index=0, format='xyz')

    def test_disjoint_graph(self):
        with self.assertRaises(AssertionError):
            generate_topology(self.atoms, cutoff_distance=1.0)

    def test_normal_graph(self):
        graph = generate_topology(self.atoms, cutoff_distance=1.6)
        self.assertEqual(len(graph.nodes), 9)
        self.assertEqual(len(graph.edges), 8)
        self.assertEqual(min(len(graph[n]) for n in graph.nodes), 1)
        self.assertEqual(max(len(graph[n]) for n in graph.nodes), 4)

    def test_data(self):
        graph = generate_topology(self.atoms, cutoff_distance=1.6, node_data=True)
        for atom, (_, node_data) in zip(self.atoms, graph.nodes(data=True)):
            self.assertEqual(atom.symbol, node_data['symbol'])
            self.assertTrue(np.allclose(atom.position, node_data['position']))
