import numpy as np
import torch.utils.data
import torch_geometric

from molgym.graph_categorical import GraphCategoricalDistribution


class TestData(torch_geometric.data.Data):
    def __init__(
            self,
            num_nodes: int,
            edge_index: torch.Tensor,  # [2, n_edges]
    ):
        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        super().__init__(
            num_nodes=num_nodes,
            edge_index=edge_index,
        )


def test_standard():
    torch.manual_seed(1)

    graphs = [
        TestData(edge_index=torch.tensor([[0], [1]], dtype=torch.long), num_nodes=2),
        TestData(edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long), num_nodes=3),
        TestData(edge_index=torch.tensor([[], []], dtype=torch.long), num_nodes=1),
        TestData(edge_index=torch.tensor([[0], [1]], dtype=torch.long), num_nodes=2),
    ]

    data_loader = torch_geometric.data.DataLoader(
        dataset=graphs,
        batch_size=len(graphs),
        shuffle=False,
        drop_last=False,
    )

    batch = next(iter(data_loader))

    logits = torch.tensor(
        [
            # Graph 1
            1.,
            3.,
            # Graph 2
            1.,
            2.,
            1.,
            # Graph 3
            1.,
            # Graph 4
            1.,
            1.,
        ],
        dtype=torch.float,
        requires_grad=True)  # [num_nodes,]

    distr = GraphCategoricalDistribution(logits=logits, batch=batch.batch, ptr=batch.ptr)
    assert distr.num_graphs == 4
    assert distr.num_nodes == 8

    samples = distr.sample(sample_shape=torch.Size((11, )))
    assert samples.shape == (11, 4)

    log_probs = distr.log_prob(samples)
    assert samples.shape == (11, 4)

    probs = torch.exp(log_probs)
    assert all(np.isclose(prob.item(), 1.0) for prob in probs[:, 2])
    assert all(np.isclose(prob.item(), 0.5) for prob in probs[:, 3])
