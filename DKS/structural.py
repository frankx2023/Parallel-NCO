import numpy as np
from numba import njit
import torch
from DKS.model import multiply


@njit(cache=True)
def core_numbers(indptr, indices):
    n = len(indptr)-1
    degree = np.diff(indptr).astype(np.int64)
    bins = np.zeros(degree.max()+1, dtype=np.int64)
    for d in degree:
        bins[d] += 1
    start = 0
    for d in range(len(bins)):
        count = bins[d]
        bins[d] = start
        start += count
    position = np.empty(n, dtype=np.int64)
    vertices = np.empty(n, dtype=np.int64)
    for v in range(n):
        position[v] = bins[degree[v]]
        vertices[position[v]] = v
        bins[degree[v]] += 1
    for d in range(len(bins)-1, 0, -1):
        bins[d] = bins[d-1]
    bins[0] = 0
    for i in range(n):
        v = vertices[i]
        for edge in range(indptr[v], indptr[v+1]):
            u = indices[edge]
            if degree[u] > degree[v]:
                du, pu = degree[u], position[u]
                pw = bins[du]
                w = vertices[pw]
                if u != w:
                    vertices[pu], vertices[pw] = w, u
                    position[u], position[w] = pw, pu
                bins[du] += 1
                degree[u] -= 1
    return degree


def choose_anchors(degree, core, m, generator, pool_size=4096):
    b, n = degree.shape
    if not 0 < m <= n:
        raise ValueError('Number of distinct anchors must be between 1 and n')
    priorities = [core.square() / (degree+1), core, degree]
    groups = [m//2, m//4, m-m//2-m//4]
    chosen = []
    used = torch.zeros_like(degree, dtype=torch.bool)
    used_count = 0
    for priority, size in zip(priorities, groups):
        count = min(n-used_count, max(pool_size, size))
        pool = priority.masked_fill(used, -torch.inf).topk(count, dim=-1).indices
        order = torch.rand(b, count, device=degree.device, generator=generator).argsort(-1)[:, :size]
        anchors = pool.gather(1, order)
        chosen.append(anchors)
        used.scatter_(1, anchors, True)
        used_count += size
    return torch.cat(chosen, -1)


def anchor_features(adjacency, degree, core, anchors, k):
    b, n = degree.shape
    m = anchors.shape[1]
    indicators = degree.new_zeros(b, n, m)
    indicators.scatter_(1, anchors[:, None, :], 1.)
    closed = multiply(adjacency, indicators) + indicators
    common = multiply(adjacency, closed)
    d = degree[:, :, None].expand(-1, -1, m)
    c = core[:, :, None].expand_as(d)
    ad = degree.gather(1, anchors)[:, None, :].expand_as(d)
    ac = core.gather(1, anchors)[:, None, :].expand_as(d)
    scale = np.log1p(k)
    features = torch.stack((d.log1p()/scale, c.log1p()/scale,
        closed, common.log1p()/scale, common/(d+1), common/(ad+1),
        common/((d+1)*(ad+1)).sqrt(), c/(d+1),
        (c-ac)/(c+ac+1), indicators), -1)
    return features
