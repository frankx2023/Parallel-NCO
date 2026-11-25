import torch


def sample_gaussian_directions(center, mean_offsets, log_variances):
    if center.dim() != 1:
        raise ValueError(f"center must have shape [num_sets], got {tuple(center.shape)}")
    if mean_offsets.dim() != 2:
        raise ValueError(
            "mean_offsets must have shape [num_sets, num_directions], "
            f"got {tuple(mean_offsets.shape)}"
        )
    if log_variances.shape != mean_offsets.shape:
        raise ValueError(
            "log_variances and mean_offsets must have identical shapes, got "
            f"{tuple(log_variances.shape)} and {tuple(mean_offsets.shape)}"
        )
    if mean_offsets.size(0) != center.numel():
        raise ValueError(
            f"center contains {center.numel()} sets but Gaussian parameters contain "
            f"{mean_offsets.size(0)}"
        )

    center = center.to(torch.float32)
    mean_offsets = mean_offsets.to(torch.float32)
    log_variances = log_variances.to(torch.float32)

    means = center.unsqueeze(1) + mean_offsets
    variances = torch.exp(log_variances)
    standard_deviations = torch.sqrt(variances)
    epsilon = torch.randn_like(means)
    directions = means + standard_deviations * epsilon

    if not torch.isfinite(variances).all():
        raise RuntimeError("Gaussian direction variances contain non-finite values")
    if not torch.isfinite(directions).all():
        raise RuntimeError("Sampled Gaussian directions contain non-finite values")

    return (
        directions.transpose(0, 1),
        means.transpose(0, 1),
        variances.transpose(0, 1),
    )


def parallel_cardinality_lmo(directions, budget):
    if directions.dim() != 2:
        raise ValueError(
            "directions must have shape [num_directions, num_sets], "
            f"got {tuple(directions.shape)}"
        )
    if not isinstance(budget, int):
        raise TypeError(f"budget must be an int, got {type(budget).__name__}")
    if budget < 1 or budget > directions.size(1):
        raise ValueError(
            f"budget must lie in [1, {directions.size(1)}], got {budget}"
        )

    indices = torch.topk(directions, k=budget, dim=-1, largest=True, sorted=False).indices
    vertices = torch.zeros_like(directions)
    vertices.scatter_(1, indices, 1.0)
    return vertices.detach()


def maximum_coverage_costs(vertices, incidence, element_weights):
    if vertices.dim() != 2:
        raise ValueError(f"vertices must be a matrix, got {tuple(vertices.shape)}")
    if incidence.dim() != 2:
        raise ValueError(f"incidence must be a matrix, got {tuple(incidence.shape)}")
    if element_weights.dim() != 1:
        raise ValueError(
            f"element_weights must be a vector, got {tuple(element_weights.shape)}"
        )
    if vertices.size(1) != incidence.size(0):
        raise ValueError(
            f"vertices use {vertices.size(1)} sets but incidence has {incidence.size(0)} rows"
        )
    if incidence.size(1) != element_weights.numel():
        raise ValueError(
            f"incidence uses {incidence.size(1)} elements but weights contain "
            f"{element_weights.numel()} values"
        )

    covered = (vertices.to(incidence.dtype) @ incidence) > 0
    coverage_weights = covered.to(element_weights.dtype) @ element_weights
    total_weight = element_weights.sum()
    if total_weight <= 0:
        raise ValueError(f"total element weight must be positive, got {total_weight.item()}")

    costs = -coverage_weights / total_weight
    if not torch.isfinite(costs).all():
        raise RuntimeError("Maximum coverage costs contain non-finite values")
    return costs


def sparse_maximum_coverage_costs(vertices, edge_index, element_weights):
    if vertices.dim() != 2:
        raise ValueError(f"vertices must be a matrix, got {tuple(vertices.shape)}")
    if edge_index.dim() != 2 or edge_index.size(0) != 2:
        raise ValueError(f"edge_index must have shape (2, edges), got {tuple(edge_index.shape)}")
    if element_weights.dim() != 1:
        raise ValueError(f"element_weights must be a vector, got {tuple(element_weights.shape)}")
    if edge_index.size(1) == 0:
        raise ValueError("sparse maximum coverage received no edges")
    if int(edge_index[0].max().item()) >= vertices.size(1):
        raise ValueError("edge_index contains a set index outside vertices")
    if int(edge_index[1].max().item()) >= element_weights.numel():
        raise ValueError("edge_index contains an element index outside element_weights")

    selected_edges = vertices[:, edge_index[0]]
    covered_counts = torch.zeros(
        (vertices.size(0), element_weights.numel()),
        dtype=selected_edges.dtype,
        device=vertices.device,
    )
    covered_counts.scatter_add_(
        1,
        edge_index[1].unsqueeze(0).expand(vertices.size(0), -1),
        selected_edges,
    )
    covered = covered_counts > 0
    total_weight = element_weights.sum()
    if not bool((total_weight > 0).item()):
        raise ValueError("total element weight must be positive")
    costs = -(
        covered.to(element_weights.dtype) * element_weights.unsqueeze(0)
    ).sum(dim=1) / total_weight
    if not bool(torch.isfinite(costs).all().item()):
        raise RuntimeError("Sparse maximum coverage costs contain non-finite values")
    return costs


def equality_kkt_extension_gradient(vertices, vertex_costs, regularization):
    if vertices.dim() != 2:
        raise ValueError(f"vertices must be a matrix, got {tuple(vertices.shape)}")
    if vertex_costs.dim() != 1:
        raise ValueError(
            f"vertex_costs must be a vector, got {tuple(vertex_costs.shape)}"
        )
    if vertices.size(0) != vertex_costs.numel():
        raise ValueError(
            f"received {vertices.size(0)} vertices and {vertex_costs.numel()} costs"
        )
    if regularization < 0:
        raise ValueError(f"regularization must be nonnegative, got {regularization}")

    original_dtype = vertices.dtype
    R = vertices.to(torch.float64)
    costs = vertex_costs.to(torch.float64)
    num_vertices = R.size(0)

    gram = R @ R.transpose(0, 1)
    gram = gram + regularization * torch.eye(
        num_vertices,
        dtype=R.dtype,
        device=R.device,
    )

    kkt = torch.zeros(
        num_vertices + 1,
        num_vertices + 1,
        dtype=R.dtype,
        device=R.device,
    )
    kkt[:num_vertices, :num_vertices] = gram
    kkt[:num_vertices, num_vertices] = 1.0
    kkt[num_vertices, :num_vertices] = 1.0

    rhs = torch.zeros(num_vertices + 1, dtype=R.dtype, device=R.device)
    rhs[:num_vertices] = costs

    solution = torch.linalg.solve(kkt, rhs)
    gradient = R.transpose(0, 1) @ solution[:num_vertices]

    if not torch.isfinite(gradient).all():
        raise RuntimeError("KKT extension gradient contains non-finite values")
    return gradient.to(original_dtype)


def prescribed_gradient_loss(interpolation_point, gradient, reported_cost):
    if interpolation_point.shape != gradient.shape:
        raise ValueError(
            "interpolation_point and gradient must have identical shapes, got "
            f"{tuple(interpolation_point.shape)} and {tuple(gradient.shape)}"
        )
    if reported_cost.dim() != 0:
        raise ValueError(
            f"reported_cost must be a scalar tensor, got {tuple(reported_cost.shape)}"
        )

    return reported_cost.detach() + (
        (interpolation_point - interpolation_point.detach()) * gradient.detach()
    ).sum()
