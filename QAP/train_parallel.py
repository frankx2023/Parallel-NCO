#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import time

import torch

from QAP import core as base
from QAP.parallel_qap import (
    DirectionSpecificGaussianQAPModel,
    GaussianQAPDirectionModel,
    parallel_qap_training_loss,
    predict_parallel_qap,
)


def require_cuda(device_name: str) -> torch.device:
    device = base.device_from_arg(device_name)
    if device.type != "cuda":
        raise RuntimeError("the parallel QAP pipeline is GPU-only; pass --device cuda")
    return device


@torch.no_grad()
def evaluate_fixed_pool(
    model: GaussianQAPDirectionModel,
    A_pool: torch.Tensor,
    B_pool: torch.Tensor,
    *,
    batch_size: int,
    num_gaussians: int,
    samples_per_gaussian: int,
) -> tuple[float, float]:
    if batch_size <= 0:
        raise RuntimeError("evaluation batch size must be positive")
    model.eval()
    costs = []
    torch.cuda.synchronize(A_pool.device)
    start = time.perf_counter()
    for offset in range(0, A_pool.shape[0], batch_size):
        end = min(offset + batch_size, A_pool.shape[0])
        prediction = predict_parallel_qap(
            model,
            A_pool[offset:end],
            B_pool[offset:end],
            num_gaussians=num_gaussians,
            samples_per_gaussian=samples_per_gaussian,
        )
        costs.append(prediction.best_cost)
    torch.cuda.synchronize(A_pool.device)
    elapsed = time.perf_counter() - start
    mean_cost = torch.cat(costs).mean().item()
    return float(mean_cost), elapsed / A_pool.shape[0]


def build_model(args):
    return DirectionSpecificGaussianQAPModel(
        num_gaussians=args.num_gaussians, max_n=args.max_n, rproj=args.rproj,
        model_dim=args.model_dim, feat_seed=args.feat_seed, dropout=args.dropout,
        initial_log_variance=args.initial_log_variance, log_variance_scale=args.log_variance_scale)


def resolve_direction_counts(args):
    total = args.num_gaussians * args.samples_per_gaussian
    if args.num_gaussians < 1 or args.samples_per_gaussian < 1 or total < 2:
        raise ValueError('Training requires positive query counts and at least two sampled directions')
    return args.num_gaussians, args.samples_per_gaussian, total


def train(args: argparse.Namespace) -> None:
    device = require_cuda(args.device)
    if args.n <= 1:
        raise RuntimeError("n must be at least 2")
    if args.max_n is None:
        args.max_n = args.n
    if args.max_n < args.n:
        raise RuntimeError(f"max_n={args.max_n} is smaller than n={args.n}")
    if args.eval_instances <= 0:
        raise RuntimeError("eval_instances must be positive")
    num_gaussians, samples_per_gaussian, total_directions = resolve_direction_counts(args)

    base.set_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    if args.dataset == "fixed":
        required_paths = [args.train_F, args.train_positions, args.test_F, args.test_positions]
        if any(path is None for path in required_paths):
            raise RuntimeError(
                "fixed data requires --train_F, --train_positions, --test_F, and --test_positions"
            )
        training_A, training_B = base.load_flow_pos_data(args.train_F, args.train_positions, device)
        validation_A, validation_B = base.load_flow_pos_data(args.test_F, args.test_positions, device)
        if training_A.shape[1] != args.n or validation_A.shape[1] != args.n:
            raise RuntimeError(
                f"configured n={args.n} does not match fixed data shapes "
                f"train={training_A.shape}, test={validation_A.shape}"
            )
        validation_count = min(args.eval_instances, validation_A.shape[0])
        validation_A = validation_A[:validation_count]
        validation_B = validation_B[:validation_count]
        steps_per_epoch = (
            (training_A.shape[0] + args.batch_size - 1) // args.batch_size
            if args.steps_per_epoch == 0
            else args.steps_per_epoch
        )
    else:
        if args.steps_per_epoch <= 0:
            raise RuntimeError("generated data requires a positive --steps_per_epoch")
        training_A = None
        training_B = None
        steps_per_epoch = args.steps_per_epoch
        validation_A, validation_B = base.gen_qap_batch(
            batch_size=args.eval_instances,
            n=args.n,
            device=device,
            symmetric=args.symmetric,
            max_int=args.max_int,
            gen_mode=args.gen_mode,
            corr_dim=args.corr_dim,
            corr_noise=args.corr_noise,
            p=args.p,
        )

    default_run_name = f"qap{args.n}_m{num_gaussians}_e{args.epochs}"
    run_name = args.run_name or default_run_name
    checkpoint_directory = os.path.join(args.checkpoint_root, run_name)
    os.makedirs(checkpoint_directory, exist_ok=True)
    best_validation_cost = float("inf")

    print(
        f"[CONFIG] device={torch.cuda.get_device_name(device)} n={args.n} max_n={args.max_n} "
        f"m={num_gaussians} samples={samples_per_gaussian} "
        f"total_directions={total_directions} epochs={args.epochs} batch={args.batch_size} "
        f"steps_per_epoch={steps_per_epoch} dataset={args.dataset} "
        f"kkt_ridge={args.kkt_ridge} kkt_solver={args.kkt_solver}"
    )
    print(f"[MODEL] parameters={parameter_count}")

    for epoch in range(args.epochs):
        model.train()
        epoch_start = time.perf_counter()
        loss_sum = 0.0
        best_cost_sum = 0.0
        mean_cost_sum = 0.0
        variance_sum = 0.0
        unique_sum = 0.0

        if args.dataset == "fixed":
            epoch_indices = torch.randperm(training_A.shape[0], device=device)
        for step in range(steps_per_epoch):
            if args.dataset == "fixed":
                start = (step * args.batch_size) % training_A.shape[0]
                end = min(start + args.batch_size, training_A.shape[0])
                indices = epoch_indices[start:end]
                if indices.numel() == 0:
                    raise RuntimeError("fixed-data batch construction produced an empty batch")
                A = training_A.index_select(0, indices)
                B = training_B.index_select(0, indices)
            else:
                A, B = base.gen_qap_batch(
                    batch_size=args.batch_size,
                    n=args.n,
                    device=device,
                    symmetric=args.symmetric,
                    max_int=args.max_int,
                    gen_mode=args.gen_mode,
                    corr_dim=args.corr_dim,
                    corr_noise=args.corr_noise,
                    p=args.p,
                )
            result = parallel_qap_training_loss(
                model,
                A,
                B,
                num_gaussians=num_gaussians,
                samples_per_gaussian=samples_per_gaussian,
                kkt_ridge=args.kkt_ridge,
                kkt_solver=args.kkt_solver,
            )
            optimizer.zero_grad(set_to_none=True)
            result.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            loss_sum += float(result.loss.item())
            best_cost_sum += float(result.best_cost.item())
            mean_cost_sum += float(result.mean_cost.item())
            variance_sum += float(result.mean_variance.item())
            unique_sum += float(result.mean_unique_vertices.item())

        validation_cost, validation_seconds_per_instance = evaluate_fixed_pool(
            model,
            validation_A,
            validation_B,
            batch_size=args.eval_batch_size,
            num_gaussians=num_gaussians,
            samples_per_gaussian=samples_per_gaussian,
        )
        torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        denominator = float(steps_per_epoch)
        print(
            f"[E{epoch:03d}] loss={loss_sum/denominator:.6f} "
            f"train_best={best_cost_sum/denominator:.3f} train_mean={mean_cost_sum/denominator:.3f} "
            f"variance={variance_sum/denominator:.4e} unique={unique_sum/denominator:.2f} "
            f"validation_best={validation_cost:.3f} "
            f"validation_ms_per_instance={validation_seconds_per_instance*1000.0:.3f} "
            f"epoch_seconds={epoch_seconds:.2f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "validation_cost": validation_cost,
        }
        torch.save(checkpoint, os.path.join(checkpoint_directory, f"epoch_{epoch:03d}.pt"))
        if validation_cost < best_validation_cost:
            best_validation_cost = validation_cost
            torch.save(checkpoint, os.path.join(checkpoint_directory, "best.pt"))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--max_n", type=int, default=None)
    parser.add_argument(
        "--m", dest="num_gaussians",
        type=int,
        default=None,
        help="parallel query budget",
    )
    parser.add_argument(
        "--samples", dest="samples_per_gaussian",
        type=int,
        default=None,
        help="samples per predicted distribution",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_instances", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--kkt_ridge", type=float, default=1e-2)
    parser.add_argument(
        "--kkt_solver",
        choices=["batched", "classical", "woodbury"],
        default="batched",
    )

    parser.add_argument("--dataset", choices=["generated", "fixed"], default="generated")
    parser.add_argument("--train_F", default=None)
    parser.add_argument("--train_positions", default=None)
    parser.add_argument("--test_F", default=None)
    parser.add_argument("--test_positions", default=None)

    parser.add_argument("--rproj", type=int, default=16)
    parser.add_argument("--model_dim", type=int, default=128)
    parser.add_argument("--feat_seed", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--initial_log_variance", type=float, default=-2.0)
    parser.add_argument("--log_variance_scale", type=float, default=2.0)

    parser.add_argument("--gen_mode", choices=["random", "correlated"], default="random")
    parser.add_argument("--max_int", type=int, default=20)
    parser.add_argument("--symmetric", action="store_true")
    parser.add_argument("--corr_dim", type=int, default=8)
    parser.add_argument("--corr_noise", type=float, default=0.2)
    parser.add_argument("--p", type=float, default=0.7)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--checkpoint_root", default="checkpoints_parallel")
    parser.add_argument("--run_name", default=None)
    return parser


if __name__ == "__main__":
    train(build_argparser().parse_args())
