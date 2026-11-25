import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from QAP import core
from QAP.eval_parallel_qaplib import load_model, canonical_costs, checkpoint_gaussian_counts
from QAP.parallel_qap import generate_model_directions, strict_parallel_greedy_lmo
from QAP.table_data import CLASSES, qaplib_data, relative_gap


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--samples', type=int, help='Samples per Gaussian; defaults to the checkpoint setting')
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--precision', choices=['high', 'highest'], default='high')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision(args.precision)
    torch.backends.cuda.matmul.allow_tf32 = args.precision == 'high'
    model, config = load_model(str(args.checkpoint), torch.device('cuda'))
    m, samples = checkpoint_gaussian_counts(config)
    if args.samples is not None:
        samples = args.samples
    records, _ = qaplib_data()
    report = dict(m=m, samples_per_gaussian=samples,
        trials=args.trials, instances=[])

    def infer(A, B, r):
        q, _ = generate_model_directions(model, A, B, m, samples_per_gaussian=samples)
        perms = strict_parallel_greedy_lmo(q.float())
        costs = canonical_costs(A.double(), B.double(), perms, swap_ab=r['swap_ab'], use_inv=r['use_inv'])
        i = costs[0].argmin()
        return perms[0, i].cpu(), float(costs[0, i]), perms

    for r in records:
        A, B = r['A'][None].cuda(), r['B'][None].cuda()
        core.set_seed(900)
        infer(A, B, r)
        observations = []
        for trial in range(args.trials):
            core.set_seed(1000+trial)
            torch.cuda.synchronize()
            start = time.perf_counter()
            permutation, value, perms = infer(A, B, r)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
            check = canonical_costs(r['A'][None].double(), r['B'][None].double(),
                permutation[None, None], swap_ab=r['swap_ab'], use_inv=r['use_inv']).item()
            assert abs(check-value) < 1e-5 and torch.equal(permutation.sort().values, torch.arange(r['n']))
            observations.append(dict(trial=trial, cost=value, seconds=elapsed, permutation=permutation.tolist()))
        item = {k:v for k,v in r.items() if k not in ['A', 'B']}
        item.update(mean_cost=statistics.mean(o['cost'] for o in observations),
                    mean_seconds=statistics.mean(o['seconds'] for o in observations), observations=observations)
        item['mean_gap_percent'] = relative_gap(item['mean_cost'], r['known'])
        report['instances'].append(item)
        print(json.dumps({k:v for k,v in item.items() if k != 'observations'}), flush=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    means = {c:statistics.mean(r['mean_gap_percent'] for r in report['instances'] if r['class']==c) for c in CLASSES}
    report.update(class_mean_gaps=means, average_class_gap=statistics.mean(means.values()),
        instance_mean_gap=statistics.mean(r['mean_gap_percent'] for r in report['instances']),
        mean_seconds=statistics.mean(r['mean_seconds'] for r in report['instances']))
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'instances'}, indent=2))


if __name__ == '__main__':
    main()
