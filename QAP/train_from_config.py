import argparse
import json
from pathlib import Path
from QAP.train_parallel import build_argparser, train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--m', type=int)
    cli = parser.parse_args()
    cfg = json.loads(cli.config.read_text())
    if 'continuation' in cfg:
        raise ValueError('Continuation configurations are not part of the reported training setup')
    args = build_argparser().parse_args([])
    unknown = set(cfg)-set(vars(args))
    if unknown:
        raise ValueError(f'Unsupported configuration keys: {sorted(unknown)}')
    vars(args).update(cfg)
    if cli.m is not None:
        args.num_gaussians = cli.m
    args.checkpoint_root = str(cli.output.parent)
    args.run_name = cli.output.name
    if cli.output.exists():
        raise FileExistsError(cli.output)
    train(args)


if __name__ == '__main__':
    main()
