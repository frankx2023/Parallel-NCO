import networkx as nx
import numpy as np
import torch

def generated(n, count, seed):
    rng = np.random.default_rng(seed)
    arrays = []
    for i in range(count):
        family = i % 5
        if family < 2:
            a = (rng.random((n, n)) < rng.uniform(.01, .15)).astype(np.float32)
            if family == 1:
                a *= rng.choice([-1., 1.], (n, n))
        elif family == 2:
            a = nx.to_numpy_array(nx.barabasi_albert_graph(n, int(rng.integers(2, 9)), seed=int(rng.integers(2**31))), dtype=np.float32)
            if rng.random() < .5:
                a *= rng.choice([-1., 1.], (n, n))
        elif family == 3:
            xy = rng.random((n, 2))
            d = ((xy[:, None]-xy[None])**2).sum(-1)
            neighbors = np.argsort(d, axis=-1)[:, 1:6]
            a = np.zeros((n, n), dtype=np.float32)
            a[np.arange(n)[:, None], neighbors] = 1
            a = np.maximum(a, a.T)
        else:
            side = rng.integers(2, size=n)
            p = np.where(side[:, None] != side[None], rng.uniform(.015, .08), rng.uniform(0, .005))
            a = (rng.random((n, n)) < p).astype(np.float32)
        lower = np.tril(a, -1)
        arrays.append(lower+lower.T)
    return torch.from_numpy(np.stack(arrays))
