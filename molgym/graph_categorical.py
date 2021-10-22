import torch
import torch_scatter


class GraphCategoricalDistribution:
    # NOTE: [Sample (S), Batch (B), Event (E)]
    def __init__(
            self,
            logits: torch.Tensor,  # [num_nodes, ]
            batch: torch.Tensor,  # [num_nodes, ]  (nodes' graph ids, e.g.,: [0, 0, 1, 1, 1, 2])
            ptr: torch.Tensor,  # [num_graphs + 1, ] (start & end pointers of graphs, e.g., [0, 2, 5, 6])
    ):
        assert len(logits.shape) == len(batch.shape) == 1
        assert logits.shape == batch.shape
        assert len(ptr.shape) == 1

        self.num_nodes = batch.shape[0]
        self.ptr = ptr
        self.num_graphs = self.ptr.shape[0] - 1

        # Compute normalized probabilities from logits
        probs = torch_scatter.scatter_softmax(src=logits, index=batch, dim=-1)  # [num_nodes,]

        # Figure out where to each entry goes in a [num_graphs, max_graph_size] tensor
        num_nodes = logits.shape[0]
        max_graph_size = max(self.ptr[1:] - self.ptr[:-1])
        node_indices = torch.arange(start=0, end=num_nodes, dtype=torch.long)
        target_indices = node_indices - self.ptr[batch] + batch * max_graph_size

        zeroes = torch.zeros(size=(self.num_graphs * max_graph_size, ), dtype=logits.dtype)
        probs_matrix = torch.scatter(input=zeroes, dim=-1, index=target_indices, src=probs)
        probs_matrix = probs_matrix.reshape(self.num_graphs, max_graph_size)  # [num_graphs, max_graph_size]

        # Create distribution
        self.distr = torch.distributions.Categorical(probs=probs_matrix)  # [num_graphs, max_graph_size]

    # Returns indices for the large (merged) graph
    def sample(self, sample_shape=torch.Size()) -> torch.Tensor:  # [S, B=num_graphs]
        samples = self.distr.sample(sample_shape)  # [S, B=num_graphs]
        return samples + self.ptr[:-1]

    # Expects indices for the large (merged) graph
    # [S, B=num_graphs]
    def log_prob(self, value) -> torch.Tensor:
        samples = value - self.ptr[:-1]
        return self.distr.log_prob(samples)  # [S, B=num_graphs]

    def __str__(self) -> str:
        return f'{self.__class__.__name__}(num_graphs={self.num_graphs}, num_nodes={self.num_nodes})'
