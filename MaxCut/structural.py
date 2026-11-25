import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
import torch
from torch import nn
from torch.nn import functional as F

def multiply(a, h):
    return torch.sparse.mm(a, h[0])[None] if a.layout == torch.sparse_csr else a@h


def cut_values(a, spins):
    total = a.values().sum().reshape(1) if a.layout == torch.sparse_csr else a.sum((-2, -1))
    return .25*(total[:, None]-(multiply(a, spins.transpose(1, 2)).transpose(1, 2)*spins).sum(-1))


def parity_features(graphs, width, seed):
    rng = np.random.default_rng(seed)
    output = []
    for a in graphs:
        csr = sparse.csr_matrix(a)
        csr = csr.copy()
        csr.data[:] = 1
        anchors = rng.choice(csr.shape[0], width, replace=False)
        distances = shortest_path(csr, directed=False, unweighted=True, indices=anchors)
        reachable = np.isfinite(distances)
        distances[~reachable] = 0
        features = ((1-2*(distances % 2))*reachable).T.astype(np.float32)
        output.append(features)
    return torch.from_numpy(np.stack(output))


class StructuralCut(nn.Module):
    def __init__(self, architecture='filter', representation='random', width=32, layers=16, normalization='symmetric'):
        super().__init__()
        if architecture not in ('filter', 'gnn') or representation not in ('random', 'parity'):
            raise ValueError((architecture, representation))
        if normalization not in ('symmetric', 'row'):
            raise ValueError(normalization)
        self.config = dict(architecture=architecture, representation=representation, width=width, layers=layers, normalization=normalization)
        self.steps = nn.Parameter(torch.zeros(layers))
        self.mixes = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(layers)) if architecture == 'gnn' else nn.ModuleList()
        for layer in self.mixes:
            nn.init.zeros_(layer.weight)
        self.query_head = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.query_head.weight)

    def embedding(self, a, degree, features):
        denominator = (degree+1).sqrt()[..., None]
        h = F.normalize(features, dim=-1)
        for layer in range(self.config['layers']):
            message = multiply(a, h/denominator)/denominator if self.config['normalization'] == 'symmetric' else multiply(a, h)/(degree[..., None]+1)
            residual = .2*self.mixes[layer](h).tanh() if self.config['architecture'] == 'gnn' else 0.
            h = F.normalize(h-(1+.9*self.steps[layer].tanh())*message+residual, dim=-1)
        return h

    def forward(self, a, degree, features, descriptors, z, independent):
        h = self.embedding(a, degree, features)
        scales = .75+.5*descriptors.sigmoid()+.2*self.query_head(descriptors).tanh()
        q = torch.einsum('bnd,bmd->bmn', h, scales*z)+.1*independent
        spins = torch.where(q >= 0, 1., -1.).detach()
        cuts = cut_values(a, spins)
        return dict(h=h, scales=scales, queries=q, spins=spins, cuts=cuts, best=cuts.max(-1).values)
