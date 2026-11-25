import argparse
import gzip
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import requests
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'DKS/data'
SOURCES = {
    'Facebook': 'facebook_combined.txt.gz',
    'web-Stanford': 'web-Stanford.txt.gz',
    'com-Youtube': 'bigdata/communities/com-youtube.ungraph.txt.gz',
    'soc-Pokec': 'soc-pokec-relationships.txt.gz',
    'as-Skitter': 'as-skitter.txt.gz',
    'com-Orkut': 'bigdata/communities/com-orkut.ungraph.txt.gz',
    'ca-HepTh': 'ca-HepTh.txt.gz',
    'ca-CondMat': 'ca-CondMat.txt.gz',
    'com-DBLP': 'bigdata/communities/com-dblp.ungraph.txt.gz',
    'roadNet-CA': 'roadNet-CA.txt.gz',
    'wiki-Talk': 'wiki-Talk.txt.gz',
    'soc-LiveJournal1': 'soc-LiveJournal1.txt.gz',
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_graphs(n, count, seed, regime='original'):
    if regime == 'sparse_community':
        return sparse_community_graphs(n, count, seed)
    if regime != 'original':
        raise ValueError(f'Unknown graph regime: {regime}')
    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(count):
        family = i % 3
        if family == 0:
            a = rng.random((n, n)) < rng.uniform(.01, .15)
        elif family == 1:
            graph = nx.barabasi_albert_graph(n, int(rng.integers(4, 13)),
                                            seed=int(rng.integers(2**31)))
            a = nx.to_numpy_array(graph, dtype=np.float32)
        else:
            groups = int(rng.integers(3, 9))
            labels = rng.choice(groups, n, p=rng.dirichlet(np.ones(groups)))
            within = rng.uniform(.15, .65, groups)
            p = np.where(labels[:, None] == labels[None, :],
                         within[labels][:, None], rng.uniform(.001, .03))
            a = rng.random((n, n)) < p
        lower = np.tril(a, -1).astype(np.float32)
        graphs.append(lower + lower.T)
    return torch.from_numpy(np.stack(graphs))


def sparse_community_graphs(n, count, seed):
    rng = np.random.default_rng(seed)
    graphs = []
    for i in range(count):
        if i % 3 == 0:
            graph = nx.barabasi_albert_graph(n, int(rng.integers(1, 9)),
                                            seed=int(rng.integers(2**31)))
            a = nx.to_numpy_array(graph, dtype=np.float32).astype(bool)
        else:
            a = rng.random((n, n)) < rng.uniform(.001, .015)
        order = rng.permutation(n)
        cursor = 0
        for _ in range(int(rng.integers(1, 6))):
            size = int(rng.integers(10, min(80, n // 5) + 1))
            nodes = order[cursor:cursor + size]
            cursor += size
            block = rng.random((size, size)) < rng.uniform(.3, .95)
            a[np.ix_(nodes, nodes)] = block
        lower = np.tril(a, -1).astype(np.float32)
        graphs.append(lower + lower.T)
    return torch.from_numpy(np.stack(graphs))


def prepare(name):
    folder = DATA / name
    folder.mkdir(parents=True, exist_ok=False)
    url = 'https://snap.stanford.edu/data/' + SOURCES[name]
    original = folder / 'original.txt.gz'
    print(f'Downloading {name}: {url}', flush=True)
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with original.open('xb') as stream:
            for chunk in response.iter_content(2**20):
                stream.write(chunk)
    print(f'Preprocessing {name}', flush=True)
    with gzip.open(original, 'rt') as stream:
        pairs = np.loadtxt(stream, dtype=np.int32, comments='#')
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    ids, inverse = np.unique(pairs, return_inverse=True)
    pairs = inverse.reshape(-1, 2).astype(np.int32)
    del inverse
    n = len(ids)
    a = sparse.coo_matrix((np.ones(len(pairs), dtype=np.float32),
        (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
    del pairs
    a = a.maximum(a.T)
    a.setdiag(0)
    a.eliminate_zeros()
    a.data[:] = 1
    a.sort_indices()
    sparse.save_npz(folder / 'adjacency.npz', a)
    np.save(folder / 'node_ids.npy', ids)
    manifest = dict(name=name, n=n, edges=a.nnz // 2)
    (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=list(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    for name in args.datasets:
        prepare(name)
