import argparse
import json
from pathlib import Path
import subprocess
import sys

from DKS.data import ROOT, SOURCES


def worker(name, output):
    import torch
    from DKS.inference import load_ours, query_inputs
    from DKS.anchor_model import AnchorNetwork
    from DKS.structural import anchor_features
    from DKS.model import cardinality_lmo, induced_edges
    from DKS.evaluation_utils import cpu_score

    cpu, ck, trained, a, d, c = load_ours(name, ROOT / 'DKS/checkpoints/model/best.pt')
    initial_ck = torch.load(ROOT / 'DKS/checkpoints/model/initial.pt',
                            map_location='cpu', weights_only=False)
    initial = AnchorNetwork(**initial_ck['model_config']).cuda().eval()
    initial.load_state_dict(initial_ck['model'], strict=True)
    assert initial.config == trained.config and initial_ck['epoch'] == 0
    k, chunk = 100, 16
    rows = []
    with torch.inference_mode():
        for seed in [2026, 3026, 4026]:
            stream = seed + list(SOURCES).index(name) * 10000 + k
            sampled, draws = query_inputs(d, c, 1000, stream)
            best = {state: (-1, None) for state in ['untrained', 'trained']}
            boundaries = sorted(set(range(0, 1001, chunk)) | {200, 1000})
            for left, right in zip(boundaries[:-1], boundaries[1:]):
                features = anchor_features(a, d, c, sampled[:, left:right], k)
                for state, model in [('untrained', initial), ('trained', trained)]:
                    q = model.queries(features, draws[:, left:right])
                    v = cardinality_lmo(q, k)
                    scores = induced_edges(a, v)
                    j = int(scores[0].argmax())
                    score = int(scores[0, j])
                    if score > best[state][0]:
                        best[state] = score, q[0, j].topk(k).indices.cpu().numpy()
                    del q, v, scores
                del features
                if right in [200, 1000]:
                    for state, (score, selected) in best.items():
                        assert cpu_score(cpu, selected, k) == score
                        rows.append(dict(graph=name, n=cpu.shape[0], k=k, seed=seed,
                                         m=right, method=state, objective=score,
                                         selected_nodes=selected.tolist()))
                    print(name, seed, right, {s:v[0] for s,v in best.items()}, flush=True)
    (output / (name + '.json')).write_text(json.dumps(rows, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', choices=list(SOURCES))
    args = parser.parse_args()
    if args.dataset:
        worker(args.dataset, args.output)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name in SOURCES:
        subprocess.run([sys.executable, '-B', '-u', '-m', 'DKS.evaluate_learning',
                        '--dataset', name, '--output', str(args.output.resolve())], cwd=ROOT, check=True)
        rows += json.loads((args.output / (name + '.json')).read_text())
        (args.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
