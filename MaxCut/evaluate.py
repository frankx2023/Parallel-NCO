import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np
from scipy import sparse
import torch
from MaxCut.ablation import AblationCut, inputs, positional
from MaxCut.graph_io import parse_graph

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--m', type=int, nargs='+', default=[1000])
    p.add_argument('--seeds', type=int, nargs='+', default=[2026, 3026, 4026])
    p.add_argument('--groups', nargs='+', choices=['gset800', 'gset1000', 'gset_large'], default=['gset800', 'gset1000', 'gset_large'])
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model = AblationCut(**ck['model_config']).cuda().eval()
    model.load_state_dict(ck['model'], strict=True)
    if max(args.m) > model.query_bank.shape[0]:
        expanded = 2 * torch.quasirandom.SobolEngine(
            model.query_bank.shape[1], scramble=True, seed=2718).draw(max(args.m)) - 1
        assert torch.equal(expanded[:model.query_bank.shape[0]], model.query_bank.cpu())
        model.query_bank = expanded.cuda()
        model.config['max_queries'] = max(args.m)
    if len(args.m) != 1:
        raise ValueError('Evaluate one budget per command to measure independent forward passes.')
    records = []
    for folder in args.groups:
        manifest = json.loads((ROOT/'data'/folder/'manifest.json').read_text())
        for index, item in enumerate(manifest['graphs']):
            name, n, edges = item['graph'], item['n'], item['edges']
            path = ROOT/'data'/folder/name
            pairs = parse_graph(path, n, edges)
            cpu = sparse.coo_matrix((np.ones(edges, dtype=np.float32), (pairs[:, 0], pairs[:, 1])), shape=(n, n)).tocsr()
            cpu = cpu+cpu.T
            a = torch.sparse_csr_tensor(torch.from_numpy(cpu.indptr).cuda(),
                torch.from_numpy(cpu.indices).cuda(), torch.from_numpy(cpu.data).cuda(), size=cpu.shape)
            degree = torch.from_numpy(np.diff(cpu.indptr).astype(np.float32)).cuda()[None]
            for seed in args.seeds:
                start = time.perf_counter()
                pa = positional([cpu], model, 900000+index+seed-2026, 'cuda')
                torch.cuda.synchronize()
                preprocessing = time.perf_counter()-start
                if seed == args.seeds[0]:
                    for _ in range(args.warmup):
                        warm_inputs = inputs(model, 1, n, max(args.m),
                            torch.Generator(device='cuda').manual_seed(seed+index), 'cuda', pa)
                        warm_output = model(a, degree, *warm_inputs)
                    if args.warmup:
                        del warm_inputs, warm_output
                    torch.cuda.synchronize()
                start = time.perf_counter()
                data = inputs(model, 1, n, max(args.m),
                    torch.Generator(device='cuda').manual_seed(seed+index), 'cuda', pa)
                out = model(a, degree, *data)
                solutions = []
                for m in args.m:
                    chosen = out['cuts'][0, :m].argmax()
                    spins = out['spins'][0, chosen].cpu().numpy().astype(np.int8)
                    solutions.append((m, spins, float(out['cuts'][0, chosen])))
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                for m, spins, reported in solutions:
                    cut = int(np.count_nonzero(spins[pairs[:, 0]] != spins[pairs[:, 1]]))
                    assert cut == reported
                    row = dict(graph=name, n=n, group=str(n) if n in (800,1000,2000) else '3000+',
                        m=m, seed=seed, cut=cut, forward_m=max(args.m), forward_seconds=elapsed,
                        preprocessing_seconds=preprocessing, solution=spins.tolist())
                    records.append(row)
                    print(json.dumps({k:v for k,v in row.items() if k != 'solution'}), flush=True)
    summary = [dict(group=group, m=m,
        mean_cut=statistics.mean(r['cut'] for r in records if r['group']==group and r['m']==m),
        mean_forward_seconds=statistics.mean(r['forward_seconds'] for r in records if r['group']==group and r['m']==m))
        for group in sorted(set(r['group'] for r in records)) for m in args.m]
    args.output.write_text(json.dumps(dict(summary=summary, records=records), indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
