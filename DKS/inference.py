from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def load_ours(dataset, checkpoint_path):
    import numpy as np
    from scipy import sparse
    import torch
    from DKS.anchor_model import AnchorNetwork
    from DKS.structural import core_numbers

    torch.set_num_threads(4)
    folder = ROOT / 'DKS/data' / dataset
    adjacency = sparse.load_npz(folder / 'adjacency.npz')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = AnchorNetwork(**checkpoint['model_config']).cuda().eval()
    model.load_state_dict(checkpoint['model'])
    if model.config['noise'] != 'none':
        raise RuntimeError('Unsupported checkpoint architecture')
    core = core_numbers(adjacency.indptr, adjacency.indices)
    a = torch.sparse_csr_tensor(torch.from_numpy(adjacency.indptr).cuda(),
        torch.from_numpy(adjacency.indices).cuda(), torch.from_numpy(adjacency.data).cuda(),
        size=adjacency.shape)
    d = torch.from_numpy(np.diff(adjacency.indptr).astype(np.float32)).cuda()[None]
    c = torch.from_numpy(core.astype(np.float32)).cuda()[None]
    return adjacency, checkpoint, model, a, d, c

def query_inputs(d, c, m, seed):
    import torch
    from DKS.structural import choose_anchors
    generator = torch.Generator(device=d.device).manual_seed(seed)
    anchors = choose_anchors(d, c, m, generator)
    anchors = anchors[:, torch.randperm(m, generator=generator, device=d.device)]
    draws = torch.randn(1, m, 6, generator=generator, device=d.device)
    return anchors, draws
