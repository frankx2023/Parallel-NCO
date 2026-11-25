import torch
from QAP import core as base
from QAP.parallel_qap import DirectionSpecificGaussianQAPModel

def checkpoint_gaussian_counts(args):
    return int(args['num_gaussians']), int(args['samples_per_gaussian'])

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    args = checkpoint['args']
    model = DirectionSpecificGaussianQAPModel(
        num_gaussians=int(args['num_gaussians']), max_n=int(args['max_n']),
        rproj=int(args['rproj']), model_dim=int(args['model_dim']),
        feat_seed=int(args['feat_seed']), dropout=float(args['dropout']),
        initial_log_variance=float(args['initial_log_variance']),
        log_variance_scale=float(args['log_variance_scale'])).to(device)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    return model.eval(), args

def canonical_costs(
    A: torch.Tensor,
    B: torch.Tensor,
    permutations: torch.Tensor,
    *,
    swap_ab: bool,
    use_inv: bool,
) -> torch.Tensor:
    batch_size, num_directions, n = permutations.shape
    if use_inv:
        evaluated_permutations = base.inv_perm(permutations.reshape(-1, n)).view(
            batch_size, num_directions, n
        )
    else:
        evaluated_permutations = permutations
    if swap_ab:
        return base.qap_cost_multi(B, A, evaluated_permutations)
    return base.qap_cost_multi(A, B, evaluated_permutations)
