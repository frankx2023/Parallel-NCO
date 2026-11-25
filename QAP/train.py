import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('QAP/checkpoints/m5000_base/config.json'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--m', type=int, default=5000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-B', '-u', '-m', 'QAP.train_from_config',
                    '--config', str(args.config), '--m', str(args.m), '--output', str(args.output/'base')], check=True)
    subprocess.run([sys.executable, '-B', '-u', '-m', 'QAP.optimize_qap32',
                    '--source', str(args.output/'base/best.pt'), '--m', str(args.m),
                    '--output', str(args.output/'selected')], check=True)


if __name__ == '__main__':
    main()
