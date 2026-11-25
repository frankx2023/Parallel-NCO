import math
import torch
from torch import nn
from DKS.model import cardinality_lmo, induced_edges
from DKS.structural import anchor_features
from MaxCut.kkt import batched_constrained_extension


def anchor_loss(adjacency, output, k, objective='kkt'):
    if objective != 'kkt':
        raise ValueError('This release contains the selected pure-KKT loss only')
    q = output['queries']
    point = q.mean(1)
    result = batched_constrained_extension(output['vertices'], point.detach(),
        -output['scores']/(k*(k-1)/2), ridge=.1)
    proxy = (point*result['gradient']).sum(-1).mean()
    loss = q.sum()*0 + proxy-proxy.detach()+result['value'].mean()
    return loss, dict(kkt_residual=result['residual'])

class AnchorNetwork(nn.Module):
    def __init__(self, architecture='mlp', noise='none', hidden=32, prior_strength=4.):
        super().__init__()
        if architecture != 'mlp' or noise != 'none':
            raise ValueError('Unsupported checkpoint architecture')
        self.config = dict(architecture=architecture, noise=noise, hidden=hidden, prior_strength=prior_strength)
        self.net = nn.Sequential(nn.Linear(10, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden//2), nn.SiLU(), nn.Linear(hidden//2, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def queries(self, features, noise_draws=None, noise_scale=1.):
        raw = self.net(features)
        q = self.config['prior_strength']*features[..., 2] + features[..., 4] + raw[..., 0].tanh()
        return q.transpose(1, 2)

    def forward(self, adjacency, degree, core, anchors, k, generator, noise_scale=1.):
        features = anchor_features(adjacency, degree, core, anchors, k)
        q = self.queries(features)
        v = cardinality_lmo(q, k)
        scores = induced_edges(adjacency, v)
        return dict(queries=q, vertices=v, scores=scores, best=scores.max(-1).values)

