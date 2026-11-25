import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def summarize(job, path):
    value = json.loads(path.read_text())
    kind = job['kind']
    if kind == 'mc':
        return dict(coverage=value['raw_coverage'], time_s=value['time_s'])
    if kind == 'qap':
        return {key: value[key] for key in
                ['class_mean_gaps', 'average_class_gap', 'instance_mean_gap', 'mean_seconds']}
    if kind == 'maxcut':
        references = json.loads((ROOT/'MaxCut/data/reference_cuts.json').read_text())
        rows = value['records']
        result = {}
        for group in list(dict.fromkeys(r['group'] for r in rows)) + ['all']:
            selected = [r for r in rows if group == 'all' or r['group'] == group]
            result[group] = dict(
                cut=statistics.mean(r['cut'] for r in selected),
                gap_percent=statistics.mean(100*(1-r['cut']/references[r['graph']]) for r in selected),
                time_s=statistics.mean(r['forward_seconds'] for r in selected))
        return result
    if kind == 'dks':
        normalizers = json.loads((ROOT/'DKS/normalizers.json').read_text())
        return dict(edges=statistics.mean(r['edges'] for r in value),
                    normalized_objective=statistics.mean(r['edges']/normalizers[r['dataset']] for r in value),
                    time_s=statistics.mean(r['time_s'] for r in value))
    if kind == 'dks_learning':
        result = {}
        for method, m in dict.fromkeys((r['method'], r['m']) for r in value):
            rows = [r for r in value if (r['method'], r['m']) == (method, m)]
            result[f'{method}_m{m}'] = dict(edges=statistics.mean(r['objective'] for r in rows))
        return result
    raise ValueError(kind)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problems', nargs='+', choices=['MC', 'QAP', 'MaxCut', 'DKS'],
                        default=['MC', 'QAP', 'MaxCut', 'DKS'])
    parser.add_argument('--suites', nargs='+', choices=['main', 'budget', 'learning', 'large'],
                        default=['main'])
    parser.add_argument('--output', type=Path, default=Path('runs/main_tables'))
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    jobs = [j for j in json.loads((ROOT/'experiments.json').read_text())
            if j['problem'] in args.problems and set(j['suites']) & set(args.suites)]
    output = args.output.resolve()
    if args.list:
        for job in jobs:
            print(job['name'] + ': ' + ' '.join([sys.executable, '-B', '-u', '-m', job['module'], *job['argv']]))
        return
    output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for job in jobs:
        directory = job['kind'] in ('dks', 'dks_learning')
        target = output/(job['name'] if directory else job['name']+'.json')
        command = [sys.executable, '-B', '-u', '-m', job['module'], *job['argv'], '--output', str(target)]
        print('[START]', job['name'], flush=True)
        with (output/(job['name']+'.log')).open('w') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        row = dict(experiment=job['name'], problem=job['problem'],
                   results=summarize(job, target/'results.json' if directory else target))
        summaries.append(row)
        (output/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
        print(json.dumps(row), flush=True)
    print('[COMPLETE] Results:', output, flush=True)


if __name__ == '__main__':
    main()
