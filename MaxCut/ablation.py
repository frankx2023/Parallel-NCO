import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
import torch
from MaxCut.exploration import ExploreCut, BASE
from MaxCut.structural import cut_values, parity_features
from MaxCut.kkt import batched_constrained_extension


def loss_function(a, out, mode='kkt', point_mode='mean'):
    if mode != 'kkt' or point_mode != 'mean':
        raise ValueError('This release contains the selected pure-KKT loss only')
    q = out['queries']
    scale = a.abs().sum((-2, -1))/2
    point = q.mean(1)
    result = batched_constrained_extension(out['spins'], point.detach(),
                                       -out['cuts']/scale[:, None], ridge=.1)
    proxy = (point*result['gradient']).sum(-1).mean()
    loss = q.sum()*0 + proxy-proxy.detach()+result['value'].mean()
    diagnostics = dict(kkt_residual=result['residual'],
        support=result['support'].float().mean().item(),
        injected_gradient_norm=result['gradient'].norm(dim=-1).mean().item())
    return loss, diagnostics

class AblationCut(ExploreCut):
    def __init__(self, architecture='gnn', representation='parity',
                 width=32, layers=64, normalization='row', max_queries=1000):
        if architecture != 'gnn' or representation != 'parity':
            raise ValueError((architecture, representation))
        if layers % 4 or normalization != 'row':
            raise ValueError('this ablation uses row normalization and depths divisible by four')
        super().__init__(architecture, 'parity' if representation == 'distance' else representation,
                         width, layers, normalization)
        self.config.update(representation=representation, max_queries=max_queries)
        latent = width*(4 if architecture == 'multiscale' else 1)
        bank = 2*torch.quasirandom.SobolEngine(2*latent, scramble=True, seed=2718).draw(max_queries)-1
        self.register_buffer('query_bank', bank)

    def forward(self, a, degree, features, descriptors, z, independent):
        h = self.embedding(a, degree, features)
        if self.config['architecture'] == 'multiscale':
            b, m, _ = descriptors.shape
            descriptors = descriptors.reshape(b, m, 4, self.config['width'])
        scales = .75+.5*descriptors.sigmoid()+.2*self.query_head(descriptors).tanh()
        if self.config['architecture'] == 'multiscale':
            scales = scales.flatten(-2)
        q = torch.einsum('bnd,bmd->bmn', h, scales*z)
        spins = torch.where(q >= 0, 1., -1.).detach()
        cuts = cut_values(a, spins)
        return dict(h=h, scales=scales, queries=q, spins=spins, cuts=cuts, best=cuts.max(-1).values)


def positional(arrays, model, seed, device):
    representation = model.config['representation']
    width = model.config['width']//2
    if representation == 'random':
        return None
    if representation == 'parity':
        return parity_features(arrays, width, seed).to(device)
    rng = np.random.default_rng(seed)
    output = []
    for a in arrays:
        csr = sparse.csr_matrix(a).copy()
        csr.data[:] = 1
        anchors = rng.choice(csr.shape[0], width, replace=False)
        distance = shortest_path(csr, directed=False, unweighted=True, indices=anchors)
        finite = np.isfinite(distance)
        distance[~finite] = 0
        frequency = np.linspace(.5, 1, width)[:, None]*np.pi
        output.append((np.cos(distance*frequency)*finite).T.astype(np.float32))
    return torch.from_numpy(np.stack(output)).to(device)


def inputs(model, b, n, m, generator, device, parity=None, training=False):
    width = model.config['width']
    features = torch.randn(b, n, width, generator=generator, device=device)
    if model.config['representation'] != 'random':
        features = torch.cat((parity, features[..., width//2:]), -1)
    latent = width*(4 if model.config['architecture'] == 'multiscale' else 1)
    if m > len(model.query_bank):
        raise ValueError('m exceeds the query bank size')
    indices = torch.randperm(len(model.query_bank), generator=generator, device=device)[:m] if training else torch.arange(m, device=device)
    descriptors, z = model.query_bank[indices].split(latent, -1)
    descriptors = descriptors[None].expand(b, -1, -1)
    z = z[None].expand(b, -1, -1)
    independent = None
    return features, descriptors, z, independent
