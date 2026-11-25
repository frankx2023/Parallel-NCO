import io
import numpy as np

def parse_graph(path, n, edges):
    stream = io.StringIO(path.read_text())
    assert tuple(map(int, stream.readline().split())) == (n, edges)
    array = np.loadtxt(stream, dtype=np.int64)
    assert array.shape == (edges, 3)
    assert np.all(array[:, 2] == 1), 'This selection must be unweighted.'
    pairs = array[:, :2] - 1
    assert pairs.min() >= 0 and pairs.max() < n
    assert np.all(pairs[:, 0] != pairs[:, 1])
    assert len(np.unique(np.sort(pairs, axis=1), axis=0)) == edges
    return pairs
