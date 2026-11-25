import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy import sparse
import torch

from DKS.data import fresh_graphs, file_hash
from DKS.structural import core_numbers, choose_anchors
from DKS.anchor_model import AnchorNetwork, anchor_loss


MODEL = {'model': ('mlp', 'none', 'kkt')}


def generated_graphs(n, count, seed, regime):
    if regime == 'sparse_community':
        return fresh_graphs(n, count, seed, regime)
    if regime != 'mixed_hard':
        raise ValueError(regime)
    out = fresh_graphs(n, count, seed, 'sparse_community').numpy()
    rng = np.random.default_rng(seed+1000000)
    original = fresh_graphs(n, (count+3)//4, seed+2000000, 'original').numpy()
    for i in range(count):
        if i % 4 == 0:
            out[i] = original[i//4]
        elif i % 4 >= 2:
            a = rng.random((n, n)) < rng.uniform(.0005, .01)
            for _ in range(int(rng.integers(2, 7))):
                size = int(rng.integers(15, min(220, n//2)))
                nodes = rng.choice(n, size, replace=False)
                a[np.ix_(nodes, nodes)] |= rng.random((size, size)) < rng.uniform(.2, .98)
            for hub in rng.choice(n, int(rng.integers(2, 12)), replace=False):
                neighbors = rng.choice(n, int(rng.integers(n//5, n//2)), replace=False)
                a[hub, neighbors] = True
                a[neighbors, hub] = True
            lower = np.tril(a, -1).astype(np.float32)
            out[i] = lower+lower.T
    return torch.from_numpy(out)


def cores_for(graphs):
    cores = []
    for graph in graphs.numpy():
        a = sparse.csr_matrix(graph)
        cores.append(core_numbers(a.indptr, a.indices))
    return torch.from_numpy(np.stack(cores).astype(np.float32))


@torch.inference_mode()
def validate(model, graphs, cores, budgets, batch, m):
    model.eval()
    rows = []
    for k in budgets:
        generator = torch.Generator(device='cuda').manual_seed(19001+k)
        scores = []
        for start in range(0, len(graphs), batch):
            a, c = graphs[start:start+batch], cores[start:start+batch]
            d = a.sum(-1)
            anchors = choose_anchors(d, c, m, generator)
            out = model(a, d, c, anchors, k, generator)
            scores.extend(out['best'].cpu().tolist())
        rows.append(dict(k=k, edges=float(np.mean(scores)), density=float(np.mean(scores))*2/(k*(k-1))))
    return float(np.mean([r['density'] for r in rows])), rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=500)
    p.add_argument('--m', type=int, default=32)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--graphs-per-epoch', type=int, default=128)
    p.add_argument('--validation-graphs', type=int, default=24)
    p.add_argument('--validation-n', type=int, default=1000)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--budgets', type=int, nargs='+', default=[20, 50, 100, 200])
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--prior-strength', type=float, default=4.)
    p.add_argument('--regime', choices=['sparse_community', 'mixed_hard'], default='mixed_hard')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    val_cpu = generated_graphs(args.validation_n, args.validation_graphs, 900001+args.seed, args.regime)
    val_cores = cores_for(val_cpu).cuda()
    np.savez_compressed(args.output/'validation.npz', adjacency=val_cpu.numpy())
    val = val_cpu.cuda()
    config = vars(args) | dict(output=str(args.output),
        validation_hash=file_hash(args.output/'validation.npz'),
        checkpoint_selection='synthetic validation mean density only; no SNAP access',
        queries='m distinct anchor-conditioned queries; one sample each; shared conditional network supports variable inference m',
        prior='mean = prior_strength*closed_neighborhood + common_neighbors/(degree+1) + tanh(neural_residual)',
        local_improvement=False,
        source_hashes={name: file_hash(Path(__file__).parent/name) for name in ['train_anchor.py', 'anchor_model.py', 'structural.py', 'model.py']})
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    models, optimizers, histories, bests = {}, {}, {}, {}

    def save(name, filename, epoch, score):
        torch.save(dict(model=models[name].state_dict(), model_config=models[name].config,
            config=config, epoch=epoch, validation=score), args.output/name/filename)

    for name in MODEL:
        (args.output/name).mkdir()
        torch.manual_seed(args.seed)
        arch, noise, _ = MODEL[name]
        models[name] = AnchorNetwork(architecture=arch, noise=noise, prior_strength=args.prior_strength).cuda()
        optimizers[name] = torch.optim.AdamW(models[name].parameters(), lr=args.lr, weight_decay=1e-5)
        score, details = validate(models[name], val, val_cores, args.budgets, args.batch, args.m)
        histories[name] = [dict(epoch=0, validation_density=score, validation=details)]
        bests[name] = score
        save(name, 'initial.pt', 0, score)
        save(name, 'best.pt', 0, score)
        print(json.dumps(histories[name][-1]), flush=True)
    rng = np.random.default_rng(100+args.seed)
    for epoch in range(1, args.epochs+1):
        start = time.perf_counter()
        graphs = generated_graphs(args.n, args.graphs_per_epoch, 50000+epoch+args.seed*100000, args.regime)
        cores = cores_for(graphs)
        records = {name: [] for name in MODEL}
        for offset in range(0, len(graphs), args.batch):
            a, c = graphs[offset:offset+args.batch].cuda(), cores[offset:offset+args.batch].cuda()
            d = a.sum(-1)
            k = int(rng.choice(args.budgets))
            seed = 100000*epoch+offset+args.seed
            gen = torch.Generator(device='cuda').manual_seed(seed)
            anchors = choose_anchors(d, c, args.m, gen)
            for name, model in models.items():
                model.train()
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                out = model(a, d, c, anchors, k, torch.Generator(device='cuda').manual_seed(seed+1))
                loss, diagnostics = anchor_loss(a, out, k)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                records[name].append(dict(loss=loss.item(), density=out['best'].mean().item()*2/(k*(k-1)),
                    grad_norm=float(grad_norm), **diagnostics))
        for name, model in models.items():
            score, details = validate(model, val, val_cores, args.budgets, args.batch, args.m)
            row = dict(epoch=epoch, validation_density=score, validation=details,
                **{key: float(np.mean([r[key] for r in records[name]])) for key in records[name][0]})
            histories[name].append(row)
            if score > bests[name]:
                bests[name] = score
                save(name, 'best.pt', epoch, score)
            save(name, 'last.pt', epoch, score)
            (args.output/name/'history.json').write_text(json.dumps(histories[name], indent=2))
            print(json.dumps(dict(seconds=time.perf_counter()-start, **row)), flush=True)


if __name__ == '__main__':
    main()
