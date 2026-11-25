from dataclasses import dataclass
import torch


@dataclass
class ConstrainedResult:
    alpha: torch.Tensor
    gradient: torch.Tensor
    value: torch.Tensor
    iterations: int
    support: int
    unique_vertices: int
    primal_residual: float
    dual_residual: float
    complementarity_residual: float


@torch.no_grad()
def constrained_extension(vertices, point, costs, ridge, *, tolerance=1e-7,
                      max_iterations=10000):
    if ridge <= 0:
        raise ValueError('constrained projection requires explicit positive ridge')
    if vertices.ndim != 2 or point.shape != (vertices.shape[1],) or costs.shape != (vertices.shape[0],):
        raise ValueError('expected vertices [m,d], point [d], costs [m]')
    R0=vertices.double(); x=point.double(); f0=costs.double()
    R,inverse,counts=torch.unique(R0,dim=0,return_inverse=True,return_counts=True)
    u=R.shape[0]
    first=torch.full((u,),R0.shape[0],dtype=torch.long,device=R.device)
    first.scatter_reduce_(0,inverse,torch.arange(len(R0),device=R.device),reduce='amin')
    f=f0[first]
    if not torch.equal(f[inverse],f0):
        raise ValueError('identical vertices must have identical costs')
    diagonal=ridge/counts.double()
    b=R@x
    alpha=torch.zeros(u,dtype=R.dtype,device=R.device)
    start=(.5*(R.square().sum(1)+diagonal)-b).argmin()
    alpha[start]=1.
    free=alpha.bool()

    def system(index):
        selected=R[index]
        H=selected@selected.T+torch.diag(diagonal[index])
        size=len(index)
        K=torch.zeros(size+1,size+1,dtype=R.dtype,device=R.device)
        K[:size,:size]=H;K[:size,size]=1.;K[size,:size]=1.
        return K

    for iteration in range(1,max_iterations+1):
        index=free.nonzero().flatten()
        K=system(index)
        solution=torch.linalg.solve(K,torch.cat((b[index],b.new_ones(1))))
        candidate=solution[:-1]
        if bool((candidate<0).any()):
            current=alpha[index]
            direction=candidate-current
            decreasing=direction<0
            ratios=torch.full_like(current,torch.inf)
            ratios[decreasing]=-current[decreasing]/direction[decreasing]
            step,blocking=ratios.min(0)
            alpha[index]=current+step*direction
            alpha[index[blocking]]=0.
            free[index[blocking]]=False
            continue
        alpha.zero_();alpha[index]=candidate
        multiplier=solution[-1]
        dual=R@(R.T@alpha)-b+diagonal*alpha+multiplier
        violation=dual.masked_fill(free,torch.inf)
        value,entering=violation.min(0)
        if value.item() < -tolerance:
            free[entering]=True
            continue
        break
    else:
        raise RuntimeError(f'constrained QP did not converge in {max_iterations} support updates')

    primal=max(abs(alpha.sum().item()-1.),max(0.,-alpha.min().item()))
    dual_residual=max(0.,-dual.min().item())
    complementarity=(alpha*dual).abs().max().item()
    if max(primal,dual_residual,complementarity)>tolerance:
        raise RuntimeError(f'KKT certificate failed: primal={primal}, dual={dual_residual}, complementarity={complementarity}')
    support=(alpha>0).nonzero().flatten()
    K=system(support)
    adjoint=torch.linalg.solve(K.T,torch.cat((f[support],f.new_zeros(1))))[:-1]
    gradient=R[support].T@adjoint
    original_alpha=alpha[inverse]/counts[inverse]
    return ConstrainedResult(alpha=original_alpha,gradient=gradient.to(vertices.dtype),
        value=(original_alpha*f0).sum().to(vertices.dtype),iterations=iteration,
        support=int((original_alpha>0).sum().item()),unique_vertices=u,
        primal_residual=primal,dual_residual=dual_residual,
        complementarity_residual=complementarity)
