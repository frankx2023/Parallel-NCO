from __future__ import annotations
import dataclasses
import gzip
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

def _die(msg: str) -> None:
    raise RuntimeError(msg)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def device_from_arg(device: str) -> torch.device:
    d = device.lower()
    if d == "cpu":
        return torch.device("cpu")
    if d.startswith("cuda"):
        if not torch.cuda.is_available():
            _die("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device(device)
    return torch.device(device)


_INT_RE = re.compile(r"[-+]?\d+")


def _read_text_or_gzip(path: str) -> str:
    if path.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read()
    with open(path, "rt", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_qaplib_dat(path: str, *, strict: bool = True) -> Tuple[int, torch.Tensor, torch.Tensor]:
    txt = _read_text_or_gzip(path)
    nums = [int(x) for x in _INT_RE.findall(txt)]
    if len(nums) < 1:
        _die(f"Failed to parse any integers from {path}")
    n = nums[0]
    need = 1 + 2 * n * n
    if len(nums) < need:
        msg = f"{path}: expected at least {need} integers (n + A + B), got {len(nums)}."
        if strict:
            _die(msg)
        raise ValueError(msg)
    off = 1
    A = torch.tensor(nums[off: off + n * n], dtype=torch.float32).reshape(n, n)
    off += n * n
    B = torch.tensor(nums[off: off + n * n], dtype=torch.float32).reshape(n, n)
    return n, A, B


def parse_qaplib_sln(path: str, n_expected: Optional[int] = None) -> Tuple[float, Optional[torch.Tensor]]:
    txt = _read_text_or_gzip(path)
    nums = [int(x) for x in _INT_RE.findall(txt)]
    if len(nums) < 2:
        _die(f"{path}: expected at least 2 integers (n and objective), got {len(nums)}.")
    n = nums[0]
    if n_expected is not None and n != n_expected:
        pass
    sol = float(nums[1])
    perm_nums = nums[2:]
    if len(perm_nums) < n:
        return sol, None
    perm_raw = torch.tensor(perm_nums[:n], dtype=torch.long)
    if (perm_raw >= 1).all() and (perm_raw <= n).all():
        perm = perm_raw - 1
    elif (perm_raw >= 0).all() and (perm_raw < n).all():
        perm = perm_raw
    else:
        return sol, None
    if not bool(perms_are_valid(perm.unsqueeze(0)).item()):
        return sol, None
    return sol, perm


def list_qaplib_instances(qaplib_dir: str) -> List[str]:
    if not os.path.isdir(qaplib_dir):
        _die(f"qaplib_dir not found: {qaplib_dir}")
    files = []
    for fn in sorted(os.listdir(qaplib_dir)):
        lo = fn.lower()
        if lo.endswith(".dat") or lo.endswith(".dat.gz"):
            files.append(os.path.join(qaplib_dir, fn))
    if not files:
        _die(f"No .dat/.dat.gz files found in {qaplib_dir}")
    return files


def find_solution_file(qapsoln_dir: str, base: str) -> Optional[str]:
    if not qapsoln_dir or not os.path.isdir(qapsoln_dir):
        return None
    candidates = [
        os.path.join(qapsoln_dir, base + ".sln"),
        os.path.join(qapsoln_dir, base + ".sln.gz"),
        os.path.join(qapsoln_dir, base + ".sol"),
        os.path.join(qapsoln_dir, base + ".sol.gz"),
        os.path.join(qapsoln_dir, base + ".SLN"),
        os.path.join(qapsoln_dir, base + ".SLN.gz"),
        os.path.join(qapsoln_dir, base + ".SOL"),
        os.path.join(qapsoln_dir, base + ".SOL.gz"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def inv_perm(perm: torch.Tensor) -> torch.Tensor:
    if perm.ndim != 2:
        _die(f"inv_perm: expected (B,n), got {perm.shape}")
    bsz, n = perm.shape
    inv = torch.empty_like(perm)
    ar = torch.arange(n, device=perm.device).view(1, n).expand(bsz, n)
    inv.scatter_(1, perm, ar)
    return inv


def perm_to_matrix(perm: torch.Tensor) -> torch.Tensor:
    if perm.ndim != 2:
        _die(f"perm_to_matrix: expected (B,n), got {perm.shape}")
    bsz, n = perm.shape
    P = torch.zeros((bsz, n, n), dtype=torch.float32, device=perm.device)
    P.scatter_(2, perm.unsqueeze(-1), 1.0)
    return P


def perms_are_valid(perms: torch.Tensor) -> torch.Tensor:
    if perms.ndim == 2:
        perm = perms
        out_shape = (perms.shape[0],)
    elif perms.ndim == 3:
        bsz, k, n = perms.shape
        perm = perms.reshape(-1, n)
        out_shape = (bsz, k)
    else:
        _die(f"perms_are_valid: expected (B,n) or (B,K,n), got {perms.shape}")

    b_flat, n = perm.shape
    if perm.dtype != torch.long:
        perm = perm.long()

    in_range = (perm >= 0) & (perm < n)

    counts = torch.zeros((b_flat, n), dtype=torch.int64, device=perm.device)
    counts.scatter_add_(1, perm, torch.ones_like(perm, dtype=torch.int64))
    unique = (counts == 1).all(dim=1)
    return (in_range.all(dim=1) & unique).view(out_shape)


def qap_cost(A: torch.Tensor, B: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    if A.ndim != 3 or B.ndim != 3 or perm.ndim != 2:
        _die(f"qap_cost: expected A,B (B,n,n) and perm (B,n), got {A.shape}, {B.shape}, {perm.shape}")
    bsz, n, n2 = A.shape
    if n != n2 or B.shape != (bsz, n, n) or perm.shape != (bsz, n):
        _die(f"qap_cost: shape mismatch A={A.shape} B={B.shape} perm={perm.shape}")

    idx_row = perm[:, :, None].expand(bsz, n, n)
    idx_col = perm[:, None, :].expand(bsz, n, n)
    B_rows = B.gather(1, idx_row)
    B_pi = B_rows.gather(2, idx_col)
    return (A * B_pi).sum(dim=(1, 2))


def qap_cost_multi(A: torch.Tensor, B: torch.Tensor, perms_bkn: torch.Tensor, *, chunk_override: Optional[int] = None) -> torch.Tensor:
    if A.ndim != 3 or B.ndim != 3 or perms_bkn.ndim != 3:
        _die(f"qap_cost_multi: expected A,B (B,n,n), perms (B,K,n), got {A.shape}, {B.shape}, {perms_bkn.shape}")
    bsz, n, n2 = A.shape
    if n != n2 or B.shape != (bsz, n, n) or perms_bkn.shape[0] != bsz or perms_bkn.shape[2] != n:
        _die(f"qap_cost_multi: shape mismatch A={A.shape} B={B.shape} perms={perms_bkn.shape}")
    K = perms_bkn.shape[1]

    chunk = chunk_override if chunk_override is not None else (64 if perms_bkn.device.type == "cpu" else 256)
    costs = []
    A_unsq = A.unsqueeze(1)
    B_unsq = B.unsqueeze(1)
    for start in range(0, K, chunk):
        end = min(start + chunk, K)
        perms_chunk = perms_bkn[:, start:end, :]
        perms_chunk = perms_chunk.to(A.device)
        idx_row = perms_chunk.unsqueeze(-1).expand(bsz, end - start, n, n)
        idx_col = perms_chunk.unsqueeze(-2).expand(bsz, end - start, n, n)
        B_rows = B_unsq.expand(bsz, end - start, n, n).gather(2, idx_row)
        B_pi = B_rows.gather(3, idx_col)
        cost_chunk = (A_unsq.expand(bsz, end - start, n, n) * B_pi).sum(dim=(2, 3))
        costs.append(cost_chunk)
    return torch.cat(costs, dim=1)


class NodeFeatureBuilder(nn.Module):
    def __init__(self, max_n: int, rproj: int, seed: int = 0, eps: float = 1e-8) -> None:
        super().__init__()
        if max_n <= 1:
            _die("NodeFeatureBuilder: max_n must be > 1")
        if rproj <= 0:
            _die("NodeFeatureBuilder: rproj must be > 0")
        self.max_n = int(max_n)
        self.rproj = int(rproj)
        self.eps = float(eps)

        gen = torch.Generator()
        gen.manual_seed(int(seed))
        R = torch.randn((self.max_n, self.rproj), generator=gen, dtype=torch.float32)
        self.register_buffer("R", R, persistent=True)

    def forward(self, M: torch.Tensor) -> torch.Tensor:
        if M.ndim != 3:
            _die(f"NodeFeatureBuilder.forward: expected (B,n,n), got {M.shape}")
        bsz, n, n2 = M.shape
        if n != n2:
            _die(f"NodeFeatureBuilder.forward: matrix must be square, got {M.shape}")
        if n > self.max_n:
            _die(f"NodeFeatureBuilder.forward: n={n} exceeds max_n={self.max_n}. Increase --max_n.")

        maxabs = M.abs().amax(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        X = M / maxabs

        diag = torch.diagonal(X, dim1=1, dim2=2)
        row_sum = X.sum(dim=2)
        col_sum = X.sum(dim=1)
        row_mean = X.mean(dim=2)
        col_mean = X.mean(dim=1)
        row_std = X.var(dim=2, unbiased=False).clamp_min(0.0).sqrt()
        col_std = X.var(dim=1, unbiased=False).clamp_min(0.0).sqrt()
        row_abs_mean = X.abs().mean(dim=2)
        col_abs_mean = X.abs().mean(dim=1)

        stats = torch.stack(
            [diag, row_sum, col_sum, row_mean, col_mean, row_std, col_std, row_abs_mean, col_abs_mean],
            dim=2,
        )

        Rn = self.R[:n, :]
        proj_row = torch.matmul(X, Rn)
        proj_col = torch.matmul(X.transpose(1, 2), Rn)

        return torch.cat([stats, proj_row, proj_col], dim=2)


def _graphsage_layer(x: torch.Tensor, adj: torch.Tensor, lin_self: nn.Linear, lin_neigh: nn.Linear) -> torch.Tensor:
    deg = adj.sum(dim=2, keepdim=True).clamp_min(1e-6)
    neigh = torch.bmm(adj, x) / deg
    return lin_self(x) + lin_neigh(neigh)


def gen_qap_batch(
    batch_size: int,
    n: int,
    *,
    device: torch.device,
    symmetric: bool,
    max_int: int,
    gen_mode: str,
    corr_dim: int,
    corr_noise: float,
    p: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n <= 1:
        _die("gen_qap_batch: n must be >= 2")
    if max_int <= 0:
        _die("gen_qap_batch: max_int must be > 0")

    if p <= 0.0 or p > 1.0:
        _die(f"gen_qap_batch: p must be in (0,1], got {p}")

    if gen_mode == "random":
        A = torch.randint(0, max_int + 1, (batch_size, n, n), device=device, dtype=torch.float32)
        B = torch.randint(0, max_int + 1, (batch_size, n, n), device=device, dtype=torch.float32)
    elif gen_mode == "correlated":
        if corr_dim <= 0:
            _die("gen_qap_batch: corr_dim must be > 0 for correlated mode")
        Z = torch.randn((batch_size, n, corr_dim), device=device)
        Za = Z + corr_noise * torch.randn_like(Z)
        Zb = Z + corr_noise * torch.randn_like(Z)
        A = Za @ Za.transpose(1, 2)
        B = Zb @ Zb.transpose(1, 2)
        A = A - A.amin(dim=(1, 2), keepdim=True)
        B = B - B.amin(dim=(1, 2), keepdim=True)
        A = A / A.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8) * float(max_int)
        B = B / B.amax(dim=(1, 2), keepdim=True).clamp_min(1e-8) * float(max_int)
    else:
        _die(f"gen_qap_batch: unknown gen_mode={gen_mode}")

    if symmetric:
        A = 0.5 * (A + A.transpose(1, 2))
        B = 0.5 * (B + B.transpose(1, 2))

    if p < 1.0:
        if symmetric:
            mask = torch.rand((batch_size, n, n), device=device) < p
            mask = mask | mask.transpose(1, 2)
        else:
            mask = torch.rand((batch_size, n, n), device=device) < p
        mask_f = mask.to(A.dtype)
        A = A * mask_f
        B = B * mask_f
    return A, B


def _pairwise_dist(pos: torch.Tensor) -> torch.Tensor:
    if pos.ndim != 3 or pos.shape[2] != 2:
        _die(f"_pairwise_dist expects (B,n,2), got {pos.shape}")
    diff = pos[:, :, None, :] - pos[:, None, :, :]
    return (diff * diff).sum(dim=-1).sqrt()


def load_flow_pos_data(
    F_path: str,
    pos_path: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not os.path.isfile(F_path) or not os.path.isfile(pos_path):
        _die(f"load_flow_pos_data: missing file(s): {F_path}, {pos_path}")
    import numpy as np

    F_np = np.load(F_path)
    pos_np = np.load(pos_path)

    if F_np.shape[0] != pos_np.shape[0]:
        _die(f"load_sawt_data: batch mismatch F={F_np.shape} pos={pos_np.shape}")
    bsz, n, n2 = F_np.shape
    if n != n2 or pos_np.shape != (bsz, n, 2):
        _die(f"load_sawt_data: unexpected shapes F={F_np.shape} pos={pos_np.shape}")

    A = torch.tensor(F_np, dtype=torch.float32, device=device)
    pos = torch.tensor(pos_np, dtype=torch.float32, device=device)
    B = _pairwise_dist(pos)

    eye = torch.eye(n, device=device).unsqueeze(0)
    A = A * (1.0 - eye)
    B = B * (1.0 - eye)

    return A, B


@dataclasses.dataclass
class QAPLibMode:
    swap_ab: bool
    use_inv: bool


_QAPLIB_MODES: List[QAPLibMode] = [
    QAPLibMode(swap_ab=False, use_inv=False),
    QAPLibMode(swap_ab=False, use_inv=True),
    QAPLibMode(swap_ab=True, use_inv=False),
    QAPLibMode(swap_ab=True, use_inv=True),
]


def qap_cost_mode(A: torch.Tensor, B: torch.Tensor, perm: torch.Tensor, mode: QAPLibMode) -> torch.Tensor:
    if mode.use_inv:
        perm = inv_perm(perm)
    if mode.swap_ab:
        return qap_cost(B, A, perm)
    return qap_cost(A, B, perm)


def detect_solution_mode(A: torch.Tensor, B: torch.Tensor, sol_obj: float, sol_perm: torch.Tensor) -> Optional[QAPLibMode]:
    if not bool(perms_are_valid(sol_perm).item()):
        return None
    target = float(sol_obj)
    for m in _QAPLIB_MODES:
        val = float(qap_cost_mode(A, B, sol_perm, m).item())
        if abs(val - target) <= 1e-6 * max(1.0, abs(target)):
            return m
    return None
