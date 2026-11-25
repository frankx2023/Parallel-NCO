import torch


@torch.no_grad()
def batched_constrained_extension(vertices, point, costs, ridge=.1, tolerance=1e-7, max_iterations=500):
    r, x, f = vertices.double(), point.double(), costs.double()
    b, m, _ = r.shape
    h = r @ r.transpose(1, 2) + ridge * torch.eye(m, dtype=r.dtype, device=r.device)
    target = (r @ x[..., None]).squeeze(-1)
    alpha = torch.full((b, m), 1 / m, dtype=r.dtype, device=r.device)
    free = torch.ones(b, m, dtype=torch.bool, device=r.device)
    running = torch.ones(b, dtype=torch.bool, device=r.device)
    rows = torch.arange(b, device=r.device)

    def system(support):
        k = r.new_zeros(b, m+1, m+1)
        k[:, :m, :m] = h * support[:, :, None] * support[:, None, :] + torch.diag_embed((~support).double())
        k[:, m, :m] = support
        k[:, :m, m] = support
        return k

    for iteration in range(max_iterations):
        k = system(free)
        rhs = torch.cat((target * free, r.new_ones(b, 1)), -1)
        solved = torch.linalg.solve(k, rhs[..., None]).squeeze(-1)
        candidate, multiplier = solved[:, :m], solved[:, m]
        negative = (candidate < 0).any(-1) & running
        direction = candidate - alpha
        ratios = torch.where(direction < 0, -alpha / direction, torch.inf)
        step, blocker = ratios.min(-1)
        step = torch.where(negative, step, 0.)
        boundary = alpha + step[:, None] * direction
        boundary[rows, blocker] = 0.
        alpha = torch.where(negative[:, None], boundary,
            torch.where(running[:, None], candidate, alpha))
        free[rows, blocker] &= ~negative
        dual = (h @ alpha[..., None]).squeeze(-1) - target + multiplier[:, None]
        violation, entering = dual.masked_fill(free, torch.inf).min(-1)
        add = running & ~negative & (violation < -tolerance)
        free[rows, entering] |= add
        done = running & ~negative & ~add
        running &= ~done
        if not bool(running.any()):
            break
    else:
        raise RuntimeError(f'batched constrained QP failed to converge after {max_iterations} iterations')

    primal = torch.maximum((alpha.sum(-1)-1).abs(), (-alpha.min(-1).values).clamp_min(0))
    dual_residual = (-dual.min(-1).values).clamp_min(0)
    complementarity = (alpha * dual).abs().max(-1).values
    residual = torch.stack((primal, dual_residual, complementarity)).max().item()
    if residual > tolerance:
        raise RuntimeError(f'batched constrained KKT certificate failed: {residual}')
    support = alpha > 0
    k = system(support)
    rhs = torch.cat((f * support, r.new_zeros(b, 1)), -1)
    z = torch.linalg.solve(k, rhs[..., None])[:, :m, 0]
    gradient = (r.transpose(1, 2) @ z[..., None]).squeeze(-1)
    return dict(alpha=alpha, gradient=gradient.to(vertices.dtype),
        value=(alpha*f).sum(-1).to(vertices.dtype), support=support.sum(-1),
        iterations=iteration+1, residual=residual)
