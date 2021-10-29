import numpy as np
import torch.utils.data
import torch_geometric
import torch_scatter

from molgym.distributions import GraphCategoricalDistribution


def generate_test_data(
        num_nodes: int,
        edge_index: torch.Tensor,  # [2, n_edges]
) -> torch_geometric.data.Data:
    assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
    return torch_geometric.data.Data(
        num_nodes=num_nodes,
        edge_index=edge_index,
    )


def test_standard():
    torch.manual_seed(1)

    graphs = [
        generate_test_data(edge_index=torch.tensor([[0], [1]], dtype=torch.long), num_nodes=2),
        generate_test_data(edge_index=torch.tensor([[0, 0], [1, 2]], dtype=torch.long), num_nodes=3),
        generate_test_data(edge_index=torch.tensor([[], []], dtype=torch.long), num_nodes=1),
        generate_test_data(edge_index=torch.tensor([[0], [1]], dtype=torch.long), num_nodes=2),
    ]

    data_loader = torch_geometric.loader.DataLoader(
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

    probs = torch_scatter.scatter_softmax(src=logits, index=batch.batch, dim=-1)  # [num_nodes,]
    distr = GraphCategoricalDistribution(probs=probs, batch=batch.batch, ptr=batch.ptr)
    assert distr.num_graphs == 4
    assert distr.num_nodes == 8

    samples = distr.sample(sample_shape=torch.Size((11, )))
    assert samples.shape == (11, 4)

    log_probs = distr.log_prob(samples)
    assert samples.shape == (11, 4)

    probs = torch.exp(log_probs)
    assert all(np.isclose(prob.item(), 1.0) for prob in probs[:, 2])
    assert all(np.isclose(prob.item(), 0.5) for prob in probs[:, 3])
