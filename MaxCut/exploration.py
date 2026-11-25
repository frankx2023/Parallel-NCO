import networkx as nx
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from MaxCut.structural import StructuralCut, multiply
from MaxCut.train_structural import generated

class ExploreCut(StructuralCut):
    def __init__(self, architecture='gnn', representation='parity', width=32,
                 layers=64, normalization='row'):
        super().__init__('gnn', representation, width, layers, normalization)
        if architecture not in ('gnn', 'multiscale', 'gated'):
            raise ValueError(architecture)
        self.config['architecture'] = architecture
        if architecture == 'multiscale':
            self.scale_logits = nn.Parameter(torch.zeros(4))
        if architecture == 'gated':
            self.gates = nn.ModuleList(nn.Linear(2*width, width) for _ in range(layers))
            for gate in self.gates:
                nn.init.zeros_(gate.weight)
                nn.init.zeros_(gate.bias)

    def embedding(self, a, degree, features):
        h = F.normalize(features, dim=-1)
        stages = []
        for layer in range(self.config['layers']):
            message = multiply(a, h)/(degree[..., None]+1)
            residual = .2*self.mixes[layer](h).tanh()
            strength = 1+.9*self.steps[layer].tanh()
            if self.config['architecture'] == 'gated':
                strength = strength*2*self.gates[layer](torch.cat((h, message), -1)).sigmoid()
            h = F.normalize(h-strength*message+residual, dim=-1)
            if (layer+1) % (self.config['layers']//4) == 0:
                stages.append(h)
        if self.config['architecture'] == 'multiscale':
            h = torch.cat([v*w.sqrt() for v, w in zip(stages, self.scale_logits.softmax(0))], -1)
        return h

    def forward(self, a, degree, features, descriptors, z, independent):
        if self.config['architecture'] != 'multiscale':
            return super().forward(a, degree, features, descriptors, z, independent)
        from MaxCut.structural import cut_values
        h = self.embedding(a, degree, features)
        b, m, _ = descriptors.shape
        descriptors = descriptors.reshape(b, m, 4, self.config['width'])
        scales = (.75+.5*descriptors.sigmoid()+.2*self.query_head(descriptors).tanh()).flatten(-2)
        q = torch.einsum('bnd,bmd->bmn', h, scales*z)+.1*independent
        spins = torch.where(q >= 0, 1., -1.).detach()
        cuts = cut_values(a, spins)
        return dict(h=h, scales=scales, queries=q, spins=spins, cuts=cuts, best=cuts.max(-1).values)


def graphs(n, count, seed, family):
    if family == 'mixed':
        return generated(n, count, seed)
    rng = np.random.default_rng(seed)
    result = []
    for i in range(count):
        kind = i % 4 if family == 'unsigned' else {'er': 0, 'regular': 1}[family]
        if kind == 0:
            a = (rng.random((n, n)) < rng.uniform(4/n, min(.15, 50/n))).astype(np.float32)
        elif kind == 1:
            d = int(rng.choice([4, 6, 10, 20]))
            a = nx.to_numpy_array(nx.random_regular_graph(d, n, seed=int(rng.integers(2**31))), dtype=np.float32)
        elif kind == 2:
            a = nx.to_numpy_array(nx.barabasi_albert_graph(n, int(rng.integers(2, 9)), seed=int(rng.integers(2**31))), dtype=np.float32)
        else:
            rows = int(np.sqrt(n))
            while n % rows:
                rows -= 1
            g = nx.grid_2d_graph(rows, n//rows, periodic=True)
            a = nx.to_numpy_array(g, dtype=np.float32)
            shortcuts = rng.random((n, n)) < rng.uniform(0, 2/n)
            a = np.maximum(a, shortcuts)
        lower = np.tril(a, -1)
        result.append(lower+lower.T)
    return torch.from_numpy(np.stack(result))


BASE = dict(architecture='gnn', representation='parity', width=32, layers=64, normalization='row')
