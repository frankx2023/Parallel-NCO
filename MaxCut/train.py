import argparse
from pathlib import Path
import torch
from MaxCut.mix512 import SETUPS, train


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.setup = 'unsigned_batch'
    torch.set_num_threads(2)
    train(args)


if __name__ == '__main__':
    main()
