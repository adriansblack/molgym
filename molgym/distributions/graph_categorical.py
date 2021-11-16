import torch


class GraphCategoricalDistribution:
    # NOTE: [Sample (S), Batch (B), Event (E)]
    def __init__(
            self,
            probs: torch.Tensor,  # [num_nodes, ]
            batch: torch.Tensor,  # [num_nodes, ]  (nodes' graph ids, e.g.,: [0, 0, 1, 1, 1, 2])
            ptr: torch.Tensor,  # [num_graphs + 1, ] (start & end pointers of graphs, e.g., [0, 2, 5, 6])
    ):
        assert len(probs.shape) == len(batch.shape) == 1
        assert probs.shape == batch.shape
        assert len(ptr.shape) == 1

        self.num_nodes = batch.shape[0]
        self.num_graphs = ptr.shape[0] - 1

        # Figure out where to each entry goes in a [num_graphs, max_graph_size] tensor
        num_nodes = probs.shape[0]
        max_graph_size = max(ptr[1:] - ptr[:-1])
        node_indices = torch.arange(start=0, end=num_nodes, dtype=torch.long, device=probs.device)
        target_indices = node_indices - ptr[batch] + batch * max_graph_size

        zeroes = torch.zeros(size=(self.num_graphs * max_graph_size, ), dtype=probs.dtype, device=probs.device)
        probs_matrix = torch.scatter(input=zeroes, dim=-1, index=target_indices, src=probs)
        probs_matrix = probs_matrix.reshape(self.num_graphs, max_graph_size)  # [num_graphs, max_graph_size]

        # Create distribution
        self.distr = torch.distributions.Categorical(probs=probs_matrix)  # [num_graphs, max_graph_size]

    # Returns indices for the individual (small) graphs
    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:  # [S, B=num_graphs]
        return self.distr.sample(sample_shape)

    # Returns indices for the individual (small) graphs
    def argmax(self) -> torch.Tensor:  # [B=num_graphs, ]
        return torch.argmax(self.distr.probs, dim=-1)

    # Expects indices for the individual (small) graphs
    def log_prob(self, value) -> torch.Tensor:
        return self.distr.log_prob(value)  # [S, B=num_graphs]

    # [B=num_graphs]
    def entropy(self) -> torch.Tensor:
        return self.distr.entropy()

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(num_graphs={self.num_graphs}, num_nodes={self.num_nodes})'
