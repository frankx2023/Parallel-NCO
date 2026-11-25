import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from MC.runtime import load_model, read_data, seed, predict


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--seed-protocol', choices=['trial', 'instance'], default='instance')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    model, cfg = load_model(args.checkpoint, 'cuda')
    model.eval()
    records = read_data(args.data)
    rows = []
    budget = torch.tensor([float(args.k)], device='cuda')

    def warm(graph, batch):
        seed(900)
        for _ in range(args.warmup):
            predict(model, graph, args.k, budget, batch)

    def observe(name, graph, weights, sets, batch, trial):
        torch.cuda.synchronize()
        start = time.perf_counter()
        selected, cost, _ = predict(model, graph, args.k, budget, batch)
        selected = selected.cpu().tolist()
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        covered = set().union(*(sets[i] for i in selected))
        raw = sum(weights[i] for i in covered)
        assert len(selected) == len(set(selected)) == args.k
        assert abs(raw/sum(weights)+float(cost)) < 1e-5
        row = dict(instance=name, k=args.k, trial=trial, coverage=float(raw), seconds=elapsed, selected=selected)
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != 'selected'}), flush=True)

    if args.seed_protocol == 'instance':
        for index, (name, graph, weights, sets) in enumerate(records):
            graph = graph.to('cuda')
            batch = torch.zeros(len(sets), dtype=torch.long, device='cuda')
            if index == 0:
                warm(graph, batch)
            for trial in range(args.trials):
                seed(args.seed + trial * 1000 + index)
                observe(name, graph, weights, sets, batch, trial)
            graph.cpu()
    else:
        for _, graph, _, _ in records:
            graph.cuda()
        batch = torch.zeros(len(records[0][3]), dtype=torch.long, device='cuda')
        warm(records[0][1], batch)
        for trial in range(args.trials):
            seed(args.seed + trial)
            for name, graph, weights, sets in records:
                observe(name, graph, weights, sets, batch, trial)
    report = dict(m=cfg[cfg['training']['method']]['num_directions'],
        seed=args.seed, trials=args.trials,
        raw_coverage=statistics.mean(r['coverage'] for r in rows),
        time_s=statistics.mean(r['seconds'] for r in rows), records=rows)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
