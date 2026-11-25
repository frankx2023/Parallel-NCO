import math
import numpy as np
import torch
from torch import amp
from torch.nn import functional as F
from torch_geometric.data import Batch as PygBatch
from constrained_kkt import constrained_extension
from MC.queries import (maximum_coverage_costs,
    parallel_cardinality_lmo, prescribed_gradient_loss, sample_gaussian_directions,
    sparse_maximum_coverage_costs)


def batch_loss(model, batch, cfg, device, epoch):
    model.last_constrained_results = []
    return _queries_batch_loss(model, batch, cfg, device, epoch)

def _coverage_costs(vertices, graph, cfg):
    backend = cfg.get("coverage_backend", "dense")
    if backend == "dense":
        return maximum_coverage_costs(vertices, graph.coverage_incidence, graph.coverage_weights)
    if backend == "sparse":
        return sparse_maximum_coverage_costs(vertices, graph.edge_index, graph.coverage_weights)
    raise ValueError(f"Unknown coverage backend: {backend}")


def _permute_red_nodes(data):
    num_red = data.x1.size(0)
    permutation = torch.randperm(num_red, device=data.x1.device)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(num_red, device=data.x1.device)

    permuted = data.clone()
    permuted.x1 = data.x1[permutation]
    permuted.coverage_incidence = data.coverage_incidence[permutation]
    permuted.edge_index = data.edge_index.clone()
    permuted.edge_index[0] = inverse[data.edge_index[0]]
    return permuted


def _parallel_training_graphs(batch, cfg, device):
    graphs = [bd.to(device) for bd, _, _ in batch]
    if cfg.get("parallel_training", {}).get("permute_red_nodes", False):
        graphs = [_permute_red_nodes(graph) for graph in graphs]
    return graphs


def _kkt_vertex_costs(costs, cfg):
    transform_cfg = cfg.get("parallel_training", {})
    transform = transform_cfg.get("vertex_cost_transform", "linear")
    if transform == "linear":
        return costs
    if transform == "exponential_coverage":
        beta = float(transform_cfg["coverage_beta"])
        if beta <= 0:
            raise ValueError(f"coverage_beta must be positive, got {beta}")
        coverage = -costs
        centered_coverage = coverage - coverage.max()
        return -torch.exp(beta * centered_coverage)
    raise ValueError(f"Unknown vertex_cost_transform: {transform}")


def _parallel_extension_loss(model, directions, vertices, costs, cfg, method):
    spec = cfg[method]
    transformed_costs = _kkt_vertex_costs(costs, cfg)
    result = constrained_extension(
        vertices, directions.mean(dim=0), transformed_costs,
        spec["kkt_regularization"],
        tolerance=spec.get("constrained_tolerance", 1e-7),
        max_iterations=spec.get("constrained_max_iterations", 10000),
    )
    model.last_constrained_results.append(result)
    return result.value.detach() + (
        (directions-directions.detach()) * result.gradient.detach().unsqueeze(0)
    ).sum(dim=1).mean()


def _direction_diversity_penalty(directions):
    if directions.dim() != 2:
        raise ValueError(
            f"directions must have shape [num_directions, num_sets], got {directions.shape}"
        )
    if directions.size(0) < 2:
        raise ValueError("direction diversity requires at least two directions")
    centered = directions - directions.mean(dim=1, keepdim=True)
    normalized = F.normalize(centered, p=2, dim=1, eps=1e-12)
    gram = normalized @ normalized.transpose(0, 1)
    off_diagonal = gram - torch.eye(
        gram.size(0), dtype=gram.dtype, device=gram.device
    )
    return off_diagonal.square().sum() / (gram.numel() - gram.size(0))


def _parallel_regularized_loss(
    base_loss,
    directions,
    cfg,
    *,
    variances=None,
):
    training_cfg = cfg.get("parallel_training", {})
    diversity_weight = float(training_cfg.get("direction_diversity_weight", 0.0))
    loss = base_loss
    if diversity_weight != 0.0:
        loss = loss + diversity_weight * _direction_diversity_penalty(directions)

    variance_weight = float(training_cfg.get("variance_floor_weight", 0.0))
    if variance_weight != 0.0:
        if variances is None:
            raise ValueError("variance_floor_weight requires Gaussian variances")
        variance_floor = float(training_cfg["variance_floor"])
        if variance_floor <= 0:
            raise ValueError(f"variance_floor must be positive, got {variance_floor}")
        loss = loss + variance_weight * torch.relu(variance_floor - variances).mean()

    variance_target_weight = float(
        training_cfg.get("variance_target_weight", 0.0)
    )
    if variance_target_weight != 0.0:
        if variances is None:
            raise ValueError("variance_target_weight requires Gaussian variances")
        variance_target = float(training_cfg["variance_target"])
        if variance_target <= 0:
            raise ValueError(
                f"variance_target must be positive, got {variance_target}"
            )
        log_variance_error = torch.log(variances) - math.log(variance_target)
        loss = loss + variance_target_weight * log_variance_error.square().mean()
    return loss


def noise_std_at_epoch(init, final, epoch, max_epoch, schedule):
    frac = min(epoch / max_epoch, 1.0)
    if schedule == "cosine":
        cos_frac = 0.5 * (1 + np.cos(np.pi * frac))
        return final + (init - final) * cos_frac
    if schedule == "linear":
        return init + frac * (final - init)
    raise ValueError(f"Unknown noise schedule: {schedule}")


def _queries_batch_loss(model, batch, cfg, device, epoch):
    pyg_list = _parallel_training_graphs(batch, cfg, device)
    pyg_batch = PygBatch.from_data_list(
        pyg_list,
        exclude_keys=["coverage_incidence", "coverage_weights", "original_graph", "k_list"],
    )

    budgets = torch.tensor([sample[1] for sample in batch], dtype=torch.float, device=device)
    sizes = torch.tensor([bd.x1.size(0) for bd in pyg_list], device=device, dtype=torch.long)
    batch_tensor = torch.repeat_interleave(
        torch.arange(len(batch), device=device, dtype=torch.long),
        sizes,
    )

    noise_cfg = cfg["training"]["noise"]
    center_noise = noise_std_at_epoch(
        noise_cfg["init_std"],
        noise_cfg["final_std"],
        epoch,
        cfg["training"]["num_epochs"],
        schedule=noise_cfg["schedule"],
    )

    with amp.autocast(device_type="cuda", enabled=False):
        center_all, mean_offsets_all, log_variances_all = model(
            pyg_batch,
            budgets,
            batch=batch_tensor,
            noise_std=center_noise,
        )

    if mean_offsets_all.size(1) != cfg["queries"]["num_directions"]:
        raise RuntimeError(
            "Gaussian direction network returned "
            f"{mean_offsets_all.size(1)} directions but config requires "
            f"{cfg['queries']['num_directions']}"
        )

    offsets = [0] + list(torch.cumsum(sizes, dim=0).tolist())
    losses = []
    best_coverages = []
    mean_coverages = []
    mean_variances = []
    unique_vertex_counts = []

    for i, (_, budget, _) in enumerate(batch):
        start, end = offsets[i], offsets[i + 1]
        center = center_all[start:end]
        mean_offsets = mean_offsets_all[start:end]
        log_variances = log_variances_all[start:end]

        directions, _, variances = sample_gaussian_directions(
            center,
            mean_offsets,
            log_variances,
        )
        vertices = parallel_cardinality_lmo(directions, int(budget))
        costs = _coverage_costs(vertices, pyg_list[i], cfg)
        loss = _parallel_extension_loss(
            model, directions, vertices, costs, cfg, "queries")
        loss = _parallel_regularized_loss(
            loss,
            directions,
            cfg,
            variances=variances,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Parallel Gaussian loss is non-finite for graph {i}")

        losses.append(loss)
        best_coverages.append(float((-costs.min()).item()))
        mean_coverages.append(float((-costs.mean()).item()))
        mean_variances.append(float(variances.mean().item()))
        unique_vertex_counts.append(int(torch.unique(vertices, dim=0).size(0)))

    model.last_recon = 0.0
    model.last_best_coverage = float(np.mean(best_coverages))
    model.last_mean_coverage = float(np.mean(mean_coverages))
    model.last_mean_variance = float(np.mean(mean_variances))
    model.last_unique_vertices = float(np.mean(unique_vertex_counts))
    return torch.stack(losses).mean()


def save_checkpoint(model, optimizer, epoch, cfg, best_val_loss, best_val_coverage, ckpt_dir, filename):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "config": cfg,
        "best_val_loss": best_val_loss,
        "best_val_coverage": best_val_coverage,
        "sharpness": model.sharpness,
    }
    torch.save(checkpoint, ckpt_dir / filename)
