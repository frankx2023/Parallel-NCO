from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from QAP import core as base


def _require_square_pair(A: torch.Tensor, B: torch.Tensor) -> tuple[int, int]:
    if A.ndim != 3 or B.ndim != 3 or A.shape != B.shape:
        raise RuntimeError(f"expected A and B with identical (batch,n,n) shapes, got {A.shape} and {B.shape}")
    batch_size, n, n2 = A.shape
    if n != n2:
        raise RuntimeError(f"expected square QAP matrices, got {A.shape}")
    return batch_size, n


class GaussianQAPDirectionModel(nn.Module):

    def __init__(
        self,
        *,
        max_n: int,
        rproj: int = 16,
        model_dim: int = 128,
        feat_seed: int = 0,
        dropout: float = 0.0,
        initial_log_variance: float = -2.0,
        log_variance_scale: float = 2.0,
    ) -> None:
        super().__init__()
        if log_variance_scale <= 0.0:
            raise RuntimeError("log_variance_scale must be positive")

        self.featA = base.NodeFeatureBuilder(max_n=max_n, rproj=rproj, seed=feat_seed)
        self.featB = base.NodeFeatureBuilder(max_n=max_n, rproj=rproj, seed=feat_seed + 999)
        in_dim = 9 + 2 * rproj
        self.in_proj = nn.Linear(in_dim, model_dim)
        self.lin_self1 = nn.Linear(model_dim, model_dim, bias=False)
        self.lin_neigh1 = nn.Linear(model_dim, model_dim, bias=False)
        self.lin_self2 = nn.Linear(model_dim, model_dim, bias=False)
        self.lin_neigh2 = nn.Linear(model_dim, model_dim, bias=False)
        self.dropout = float(dropout)

        self.mean_A = nn.Linear(model_dim, model_dim, bias=False)
        self.mean_B = nn.Linear(model_dim, model_dim, bias=False)
        self.log_variance_A = nn.Linear(model_dim, model_dim, bias=False)
        self.log_variance_B = nn.Linear(model_dim, model_dim, bias=False)
        self.initial_log_variance = nn.Parameter(torch.tensor(float(initial_log_variance)))
        self.log_variance_scale = float(log_variance_scale)

    def _encode(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(features)
        residual = h
        h = base._graphsage_layer(h, adjacency, self.lin_self1, self.lin_neigh1)
        h = F.relu(h)
        if self.dropout > 0.0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        h = base._graphsage_layer(h, adjacency, self.lin_self2, self.lin_neigh2)
        h = F.relu(h + residual)
        if self.dropout > 0.0:
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    @staticmethod
    def _remove_assignment_gauge(scores: torch.Tensor) -> torch.Tensor:
        row_mean = scores.mean(dim=-1, keepdim=True)
        col_mean = scores.mean(dim=-2, keepdim=True)
        overall_mean = scores.mean(dim=(-2, -1), keepdim=True)
        return scores - row_mean - col_mean + overall_mean

    @staticmethod
    def _direction_scores(
        left: torch.Tensor,
        right: torch.Tensor,
        head: nn.Linear,
    ) -> torch.Tensor:
        scale = math.sqrt(left.shape[-1])
        scores = torch.einsum("bid,bjd,md->bmij", left, right, head.weight) / scale
        return scores + head.bias.view(1, -1, 1, 1)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _require_square_pair(A, B)
        hA = self._encode(self.featA(A), A.clamp_min(0.0))
        hB = self._encode(self.featB(B), B.clamp_min(0.0))
        scale = math.sqrt(hA.shape[-1])

        mean = torch.einsum("bid,bjd->bij", self.mean_A(hA), self.mean_B(hB)) / scale
        mean = self._remove_assignment_gauge(mean)

        log_variance_score = (
            torch.einsum(
                "bid,bjd->bij",
                self.log_variance_A(hA),
                self.log_variance_B(hB),
            )
            / scale
        )
        log_variance = self.initial_log_variance + self.log_variance_scale * torch.tanh(log_variance_score)
        return mean, log_variance


class DirectionSpecificGaussianQAPModel(GaussianQAPDirectionModel):

    def __init__(self, *, num_gaussians: int, **kwargs) -> None:
        if num_gaussians < 1:
            raise RuntimeError("the number of Gaussian matrices must be positive")
        super().__init__(**kwargs)
        self.num_gaussians = int(num_gaussians)
        model_dim = self.mean_A.out_features
        self.mean_direction_head = nn.Linear(model_dim, self.num_gaussians)
        self.log_variance_direction_head = nn.Linear(model_dim, self.num_gaussians)
        nn.init.normal_(self.mean_direction_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.mean_direction_head.bias)
        nn.init.normal_(self.log_variance_direction_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.log_variance_direction_head.bias)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _require_square_pair(A, B)
        hA = self._encode(self.featA(A), A.clamp_min(0.0))
        hB = self._encode(self.featB(B), B.clamp_min(0.0))

        mean = self._direction_scores(
            self.mean_A(hA),
            self.mean_B(hB),
            self.mean_direction_head,
        )
        mean = self._remove_assignment_gauge(mean)

        log_variance_score = self._direction_scores(
            self.log_variance_A(hA),
            self.log_variance_B(hB),
            self.log_variance_direction_head,
        )
        log_variance = self.initial_log_variance + self.log_variance_scale * torch.tanh(
            log_variance_score
        )
        return mean, log_variance


def sample_direction_specific_gaussian_directions(
    means: torch.Tensor,
    log_variances: torch.Tensor,
    *,
    samples_per_gaussian: int,
    epsilon: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if means.ndim != 4 or log_variances.shape != means.shape:
        raise RuntimeError(
            "expected direction-specific means/log_variances (batch,m,n,n), got "
            f"{means.shape} and {log_variances.shape}"
        )
    if means.shape[-1] != means.shape[-2]:
        raise RuntimeError(f"direction-specific assignment matrices must be square, got {means.shape}")
    if samples_per_gaussian <= 0:
        raise RuntimeError("samples_per_gaussian must be positive")
    batch_size, num_gaussians, n, _ = means.shape
    expected_shape = (
        batch_size,
        num_gaussians,
        samples_per_gaussian,
        n,
        n,
    )
    if epsilon is None:
        epsilon = torch.randn(expected_shape, dtype=means.dtype, device=means.device)
    elif epsilon.shape != expected_shape:
        raise RuntimeError(f"epsilon must have shape {expected_shape}, got {epsilon.shape}")
    sampled = means.unsqueeze(2) + torch.exp(0.5 * log_variances).unsqueeze(2) * epsilon
    return sampled.reshape(batch_size, num_gaussians * samples_per_gaussian, n, n)


def generate_model_directions(model, A, B, num_gaussians, *, samples_per_gaussian, epsilon=None):
    if num_gaussians != model.num_gaussians:
        raise ValueError('The query count must match the trained output heads')
    mean, log_variance = model(A, B)
    directions = sample_direction_specific_gaussian_directions(
        mean, log_variance, samples_per_gaussian=samples_per_gaussian, epsilon=epsilon)
    return directions, log_variance


@torch.no_grad()
def strict_parallel_greedy_lmo(directions: torch.Tensor) -> torch.Tensor:
    if directions.ndim != 4:
        raise RuntimeError(f"expected directions (batch,m,n,n), got {directions.shape}")
    batch_size, num_directions, n, n2 = directions.shape
    if n != n2:
        raise RuntimeError(f"assignment directions must be square, got {directions.shape}")
    if directions.dtype != torch.float32:
        raise RuntimeError(f"strict_parallel_greedy_lmo requires float32 directions, got {directions.dtype}")

    scores = directions.reshape(batch_size * num_directions, n, n)
    flat_batch = scores.shape[0]
    row_free = torch.ones((flat_batch, n), dtype=torch.bool, device=scores.device)
    col_free = torch.ones((flat_batch, n), dtype=torch.bool, device=scores.device)
    permutation = torch.full((flat_batch, n), -1, dtype=torch.long, device=scores.device)
    row_index = torch.arange(n, device=scores.device).view(1, n).expand(flat_batch, n)
    negative_infinity = torch.tensor(-torch.inf, dtype=scores.dtype, device=scores.device)

    for _ in range(n):
        masked_scores = scores.masked_fill(~col_free[:, None, :], negative_infinity)
        selected_column = masked_scores.argmax(dim=2)
        selected_value = masked_scores.gather(2, selected_column.unsqueeze(-1)).squeeze(-1)
        selected_value = selected_value.masked_fill(~row_free, negative_infinity)

        best_value = torch.full((flat_batch, n), negative_infinity, dtype=scores.dtype, device=scores.device)
        best_value.scatter_reduce_(1, selected_column, selected_value, reduce="amax", include_self=True)
        tied_for_column = row_free & (selected_value == best_value.gather(1, selected_column))

        losing_row = torch.full_like(row_index, n)
        proposed_row = torch.where(tied_for_column, row_index, losing_row)
        best_row = torch.full((flat_batch, n), n, dtype=torch.long, device=scores.device)
        best_row.scatter_reduce_(1, selected_column, proposed_row, reduce="amin", include_self=True)
        winners = tied_for_column & (row_index == best_row.gather(1, selected_column))

        permutation[winners] = selected_column[winners]
        row_free[winners] = False
        used_columns = torch.zeros((flat_batch, n), dtype=torch.int64, device=scores.device)
        used_columns.scatter_add_(1, selected_column, winners.to(torch.int64))
        col_free &= used_columns == 0

    valid = base.perms_are_valid(permutation)
    if not bool(valid.all().item()):
        invalid = torch.nonzero(~valid, as_tuple=False).flatten().tolist()
        raise RuntimeError(f"strict parallel greedy LMO produced invalid permutations at flat indices {invalid}")
    return permutation.view(batch_size, num_directions, n)


def permutation_vertex_gram(permutations: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if permutations.ndim != 3:
        raise RuntimeError(f"expected permutations (batch,m,n), got {permutations.shape}")
    if not bool(base.perms_are_valid(permutations).all().item()):
        raise RuntimeError("permutation_vertex_gram received invalid permutations")
    matches = permutations[:, :, None, :] == permutations[:, None, :, :]
    return matches.sum(dim=3).to(dtype=dtype)


def _solve_dense_systems(
    matrix: torch.Tensor,
    rhs: torch.Tensor,
    *,
    solver: str,
) -> torch.Tensor:
    if matrix.ndim != 3 or matrix.shape[1] != matrix.shape[2]:
        raise RuntimeError(f"expected batched square matrices, got {matrix.shape}")
    if rhs.ndim != 3 or rhs.shape[0] != matrix.shape[0] or rhs.shape[1] != matrix.shape[1]:
        raise RuntimeError(f"incompatible matrix/rhs shapes: {matrix.shape} and {rhs.shape}")
    if solver == "batched":
        return torch.linalg.solve(matrix, rhs)
    if solver == "classical":
        return torch.stack(
            [torch.linalg.solve(matrix[index], rhs[index]) for index in range(matrix.shape[0])],
            dim=0,
        )
    raise RuntimeError(f"unknown dense linear solver: {solver}")


def equality_kkt_extension_gradient(
    permutations: torch.Tensor,
    costs: torch.Tensor,
    *,
    ridge: float,
    linear_solver: str = "batched",
) -> torch.Tensor:
    if permutations.ndim != 3 or costs.ndim != 2 or costs.shape[:2] != permutations.shape[:2]:
        raise RuntimeError(f"shape mismatch: permutations={permutations.shape}, costs={costs.shape}")
    if ridge <= 0.0:
        raise RuntimeError("ridge must be strictly positive so duplicate sampled vertices are well-defined")
    batch_size, num_directions, n = permutations.shape
    gram = permutation_vertex_gram(permutations, dtype=costs.dtype)
    K = torch.zeros(
        (batch_size, num_directions + 1, num_directions + 1),
        dtype=costs.dtype,
        device=costs.device,
    )
    identity = torch.eye(num_directions, dtype=costs.dtype, device=costs.device).unsqueeze(0)
    K[:, :num_directions, :num_directions] = gram + ridge * identity
    K[:, :num_directions, num_directions] = 1.0
    K[:, num_directions, :num_directions] = 1.0

    rhs = torch.zeros((batch_size, num_directions + 1, 1), dtype=costs.dtype, device=costs.device)
    rhs[:, :num_directions, 0] = costs
    dual = _solve_dense_systems(
        K.transpose(1, 2),
        rhs,
        solver=linear_solver,
    )[:, :num_directions, 0]
    if not bool(torch.isfinite(dual).all().item()):
        raise RuntimeError("KKT solve produced non-finite values")

    vertices = F.one_hot(permutations, num_classes=n).to(dtype=costs.dtype)
    return torch.einsum("bm,bmnk->bnk", dual, vertices)


def low_rank_kkt_extension_gradient(
    permutations: torch.Tensor,
    costs: torch.Tensor,
    *,
    ridge: float,
    linear_solver: str = "batched",
) -> torch.Tensor:
    if permutations.ndim != 3 or costs.ndim != 2 or costs.shape[:2] != permutations.shape[:2]:
        raise RuntimeError(f"shape mismatch: permutations={permutations.shape}, costs={costs.shape}")
    if ridge <= 0.0:
        raise RuntimeError("ridge must be strictly positive so duplicate sampled vertices are well-defined")
    if not bool(base.perms_are_valid(permutations).all().item()):
        raise RuntimeError("low_rank_kkt_extension_gradient received invalid permutations")

    batch_size, _, n = permutations.shape
    vertices = F.one_hot(permutations, num_classes=n).to(dtype=costs.dtype).flatten(2)
    V = vertices.transpose(1, 2)
    dimension = n * n
    system = V @ V.transpose(1, 2)
    system = system + ridge * torch.eye(
        dimension,
        dtype=costs.dtype,
        device=costs.device,
    ).unsqueeze(0)

    weighted_vertices = V @ costs.unsqueeze(2)
    vertex_sum = V.sum(dim=2, keepdim=True)
    solved = _solve_dense_systems(
        system,
        torch.cat((weighted_vertices, vertex_sum), dim=2),
        solver=linear_solver,
    )
    unconstrained_gradient = solved[:, :, 0]
    constraint_gradient = solved[:, :, 1]
    denominator = constraint_gradient.sum(dim=1)
    if not bool((denominator.abs() > torch.finfo(costs.dtype).eps).all().item()):
        raise RuntimeError("low-rank KKT equality multiplier has a zero denominator")
    multiplier = unconstrained_gradient.sum(dim=1) / denominator
    gradient = unconstrained_gradient - constraint_gradient * multiplier.unsqueeze(1)
    if not bool(torch.isfinite(gradient).all().item()):
        raise RuntimeError("low-rank KKT solve produced non-finite values")
    return gradient.reshape(batch_size, n, n)


def extension_gradient_for_solver(
    permutations: torch.Tensor,
    costs: torch.Tensor,
    *,
    ridge: float,
    kkt_solver: str,
) -> torch.Tensor:
    if kkt_solver in {"batched", "classical"}:
        return equality_kkt_extension_gradient(
            permutations,
            costs,
            ridge=ridge,
            linear_solver=kkt_solver,
        )
    if kkt_solver == "woodbury":
        return low_rank_kkt_extension_gradient(
            permutations,
            costs,
            ridge=ridge,
            linear_solver="classical",
        )
    raise RuntimeError(f"unknown KKT solver: {kkt_solver}")


def normalize_qap_costs(costs: torch.Tensor) -> torch.Tensor:
    if costs.ndim != 2:
        raise RuntimeError(f"expected costs (batch,m), got {costs.shape}")
    scale = costs.abs().mean(dim=1, keepdim=True)
    if not bool((scale > 0.0).all().item()):
        raise RuntimeError("cannot normalize a QAP batch with zero mean absolute sampled cost")
    return (costs - costs.mean(dim=1, keepdim=True)) / scale


def prescribed_gradient_loss(point: torch.Tensor, gradient: torch.Tensor) -> torch.Tensor:
    if point.shape != gradient.shape:
        raise RuntimeError(f"point and gradient shapes differ: {point.shape} versus {gradient.shape}")
    return (point * gradient.detach()).flatten(1).sum(dim=1).mean()


@dataclasses.dataclass
class ParallelQAPTrainingResult:
    loss: torch.Tensor
    best_cost: torch.Tensor
    mean_cost: torch.Tensor
    mean_variance: torch.Tensor
    mean_unique_vertices: torch.Tensor
    constrained_results: Optional[list] = None


def parallel_qap_training_loss(
    model: GaussianQAPDirectionModel,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    num_gaussians: int,
    samples_per_gaussian: int,
    kkt_ridge: float,
    kkt_solver: str = "batched",
) -> ParallelQAPTrainingResult:
    directions, log_variance = generate_model_directions(
        model,
        A,
        B,
        num_gaussians,
        samples_per_gaussian=samples_per_gaussian,
    )
    directions = directions.float()
    permutations = strict_parallel_greedy_lmo(directions)
    costs = base.qap_cost_multi(A.float(), B.float(), permutations).detach()
    normalized_costs = normalize_qap_costs(costs)
    differentiable_point = directions.mean(dim=1)
    unique_counts = [torch.unique(p, dim=0).size(0) for p in permutations]
    constrained_results = None
    extension_gradient = extension_gradient_for_solver(
        permutations, normalized_costs, ridge=kkt_ridge, kkt_solver=kkt_solver)
    loss = prescribed_gradient_loss(differentiable_point, extension_gradient)

    mean_unique = torch.tensor(
        unique_counts,
        device=permutations.device, dtype=torch.float32,
    ).mean()

    mean_variance = (
        log_variance.exp().mean()
        if log_variance is not None
        else torch.zeros((), dtype=directions.dtype, device=directions.device)
    )
    return ParallelQAPTrainingResult(
        loss=loss,
        best_cost=costs.min(dim=1).values.mean(),
        mean_cost=costs.mean(),
        mean_variance=mean_variance,
        mean_unique_vertices=mean_unique,
        constrained_results=constrained_results,
    )


@dataclasses.dataclass
class ParallelQAPPrediction:
    best_permutation: torch.Tensor
    best_cost: torch.Tensor
    permutations: torch.Tensor
    costs: torch.Tensor


@torch.no_grad()
def predict_parallel_qap(
    model: GaussianQAPDirectionModel,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    num_gaussians: int,
    samples_per_gaussian: int,
    epsilon: Optional[torch.Tensor] = None,
) -> ParallelQAPPrediction:
    model.eval()
    directions, _ = generate_model_directions(
        model,
        A,
        B,
        num_gaussians,
        samples_per_gaussian=samples_per_gaussian,
        epsilon=epsilon,
    )
    directions = directions.float()
    permutations = strict_parallel_greedy_lmo(directions)
    costs = base.qap_cost_multi(A.float(), B.float(), permutations)
    best_cost, best_index = costs.min(dim=1)
    batch_index = torch.arange(A.shape[0], device=A.device)
    best_permutation = permutations[batch_index, best_index]
    return ParallelQAPPrediction(
        best_permutation=best_permutation,
        best_cost=best_cost,
        permutations=permutations,
        costs=costs,
    )
