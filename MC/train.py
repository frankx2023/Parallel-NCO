import argparse
import copy
import csv
import json
from pathlib import Path
import random
import statistics
import time

import torch
import yaml

from MC.runtime import make_model, read_data, seed
from MC.train_methods import batch_loss, save_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--k', type=int, required=True)
    p.add_argument('--m', type=int)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = yaml.safe_load(args.config.read_text())
    method = cfg['training']['method']
    cfg = copy.deepcopy(cfg)
    if args.m is not None:
        cfg[method]['num_directions'] = args.m
    cfg['training']['num_epochs'] = args.epochs
    cfg['training']['data']['k_list'] = [args.k]
    if args.lr is not None:
        cfg['training']['cosine_eta_min'] *= args.lr/cfg['training']['learning_rate']
        cfg['training']['learning_rate'] = args.lr
    args.output.mkdir(parents=True, exist_ok=False)
    seed(cfg['training']['seed'])
    records = read_data(args.data)
    cfg['coverage_backend'] = 'dense' if 'coverage_incidence' in records[0][1] else 'sparse'
    n, elements = len(records[0][3]), len(records[0][2])
    cfg['training']['data'].update(red_n=n, blue_n=elements, num_graphs=len(records))
    (args.output/'config.yaml').write_text(yaml.safe_dump(cfg))
    samples = [(graph, args.k, None) for _, graph, _, _ in records]
    model = make_model(cfg).cuda()
    base_lr = cfg['training']['learning_rate']
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr,
                            weight_decay=cfg['training']['weight_decay'])
    warmup = cfg['training']['warmup_epochs']
    if args.epochs <= warmup:
        raise ValueError('Training epochs must exceed the configured warmup')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
        T_max=args.epochs-warmup, eta_min=cfg['training']['cosine_eta_min'])
    batch_size = cfg['training']['batch_size']
    with (args.output/'training.csv').open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'lr', 'surrogate_reported', 'train_best_ratio', 'unique', 'seconds'])
        for step in range(args.epochs):
            started = time.perf_counter()
            epoch = step
            model.train()
            random.shuffle(samples)
            if step < warmup:
                for group in opt.param_groups:
                    group['lr'] = base_lr*(step+1)/warmup
            losses, coverage, unique = [], [], []
            for offset in range(0, len(samples), batch_size):
                opt.zero_grad(set_to_none=True)
                batch = [(graph.clone().to('cuda'), k, extra)
                         for graph, k, extra in samples[offset:offset+batch_size]]
                loss = batch_loss(model, batch, cfg, torch.device('cuda'), step)
                loss.backward()
                opt.step()
                losses.append(loss.item())
                coverage.append(model.last_best_coverage)
                unique.append(model.last_unique_vertices)
            if step >= warmup:
                scheduler.step()
            row = [epoch, opt.param_groups[0]['lr'], statistics.mean(losses),
                   statistics.mean(coverage), statistics.mean(unique), time.perf_counter()-started]
            writer.writerow(row)
            handle.flush()
            print(json.dumps(dict(zip(['epoch', 'lr', 'loss', 'coverage', 'unique', 'seconds'], row))), flush=True)
    save_checkpoint(model, opt, epoch, cfg, None, None, args.output, 'final.pt')


if __name__ == '__main__':
    main()
