import torch

def multiply(adjacency, features):
    if adjacency.layout == torch.sparse_csr:
        return torch.sparse.mm(adjacency, features[0]).unsqueeze(0)
    return torch.bmm(adjacency, features)


def cardinality_lmo(queries, k):
    selected = queries.topk(k, dim=-1).indices
    return torch.zeros_like(queries).scatter_(-1, selected, 1.).detach()


def induced_edges(adjacency, vertices):
    return .5 * (multiply(adjacency, vertices.transpose(1, 2)).transpose(1, 2)
                 * vertices).sum(-1)
