import numpy as np

def cpu_score(adjacency, selected, k):
    assert len(selected) == k and len(np.unique(selected)) == k
    assert selected.min() >= 0 and selected.max() < adjacency.shape[0]
    return int(adjacency[selected][:, selected].sum() / 2)
