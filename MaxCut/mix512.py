import json
from pathlib import Path
import time
import networkx as nx
import numpy as np
import torch
from MaxCut import run_ablation
from MaxCut.ablation import AblationCut, BASE, positional, inputs, loss_function
from MaxCut.exploration import graphs as original_graphs
from MaxCut.data import ROOT, file_hash

SETUPS = {
    'unsigned_cycle': dict(family='unsigned', sizes=[250, 500, 1000], schedule='epoch'),
    'unsigned_batch': dict(family='unsigned', sizes=[250, 500, 1000], schedule='batch'),
    'unsigned_large': dict(family='unsigned', sizes=[500, 1000, 2000], schedule='batch'),
    'unsigned_curriculum': dict(family='unsigned', sizes=[250]*10+[500]*10+[1000]*10, schedule='epoch'),
    'mixed500': dict(family='mixed', sizes=[500], schedule='epoch'),
    'mixed_batch': dict(family='mixed', sizes=[250, 500, 1000], schedule='batch'),
    'er_batch': dict(family='er', sizes=[250, 500, 1000], schedule='batch'),
    'regular_batch': dict(family='regular', sizes=[250, 500, 1000], schedule='batch'),
    'structured500': dict(family='structured', sizes=[500], schedule='epoch'),
    'structured_batch': dict(family='structured', sizes=[250, 500, 1000], schedule='batch'),
}


def graphs(n, count, seed, family):
    if family in ('unsigned', 'mixed', 'er', 'regular'):
        return original_graphs(n, count, seed, family)
    if family != 'structured':
        raise ValueError(family)
    rng = np.random.default_rng(seed)
    arrays = []
    for index in range(count):
        kind = index % 4
        graph_seed = int(rng.integers(2**31))
        if kind == 0:
            a = nx.to_numpy_array(nx.barabasi_albert_graph(n, int(rng.integers(2, 9)), seed=graph_seed), dtype=np.float32)
        elif kind == 1:
            xy = rng.random((n, 2))
            distances = ((xy[:, None]-xy[None])**2).sum(-1)
            degree = int(rng.choice([4, 8, 12]))
            neighbors = np.argsort(distances, axis=-1)[:, 1:degree+1]
            a = np.zeros((n, n), dtype=np.float32)
            a[np.arange(n)[:, None], neighbors] = 1
            a = np.maximum(a, a.T)
        elif kind == 2:
            labels = rng.integers(0, 4, n)
            degree = rng.uniform(6, 30)
            high, low = 2*degree/n, degree/(3*n)
            if rng.random() < .5:
                high, low = low, high
            probability = np.where(labels[:, None] == labels[None], high, low)
            a = (rng.random((n, n)) < probability).astype(np.float32)
        else:
            a = nx.to_numpy_array(nx.watts_strogatz_graph(n, int(rng.choice([4, 8, 12])),
                float(rng.uniform(.02, .3)), seed=graph_seed), dtype=np.float32)
        lower = np.tril(a, -1)
        arrays.append(lower+lower.T)
    return torch.from_numpy(np.stack(arrays))


def epoch_sizes(setup, epoch):
    sizes = setup['sizes']
    if setup['schedule'] == 'epoch':
        return [sizes[(epoch-1) % len(sizes)]]*16
    if setup['schedule'] == 'batch':
        return [sizes[((epoch-1)*16+i) % len(sizes)] for i in range(16)]
    raise ValueError(setup['schedule'])


def train(args):
    args.output.mkdir(parents=True, exist_ok=False)
    setup = SETUPS[args.setup] | dict(loss='kkt', point='mean', layers=512)
    torch.manual_seed(0)
    model = AblationCut(**(BASE | dict(layers=512))).cuda()
    validation = torch.cat((graphs(500, 20, 880001, 'mixed'), graphs(500, 20, 880002, 'unsigned')))
    val_parity = positional(validation.numpy(), model, 990000, 'cuda')
    np.savez_compressed(args.output/'validation.npz', adjacency=validation.numpy())
    config = dict(output=str(args.output), setup=setup,
        epochs=30, graphs_per_epoch=64, m=16, seed=0, lr=.001, model_config=model.config,
        validation_hash=file_hash(args.output/'validation.npz'),
        sources={name: file_hash(Path(__file__).parent/name) for name in ['mix512.py', 'ablation.py', 'exploration.py', 'structural.py', 'kkt.py', 'run_ablation.py']},
        selection='unchanged 40 generated validation graphs; best-of-50 normalized cut; initial eligible')
    (args.output/'config.json').write_text(json.dumps(config, indent=2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    def save(file, epoch, score):
        torch.save(dict(model=model.state_dict(), model_config=model.config, setup=setup,
            config=config, epoch=epoch, validation=score), args.output/file)
    best = run_ablation.validate(model, validation, val_parity, 'cuda')
    save('initial.pt', 0, best)
    save('best.pt', 0, best)
    history = [dict(epoch=0, validation=best)]
    for epoch in range(1, 31):
        start = time.perf_counter()
        sizes = epoch_sizes(setup, epoch)
        batches, feature_batches, offsets = {}, {}, {}
        for n in sorted(set(sizes)):
            seed = 50000+epoch if setup['schedule'] == 'epoch' else 50000+epoch+137*n
            batches[n] = graphs(n, 4*sizes.count(n), seed, setup['family'])
            feature_batches[n] = positional(batches[n].numpy(), model, 150000+epoch, 'cuda')
            offsets[n] = 0
        model.train()
        records = []
        for batch_index, n in enumerate(sizes):
            offset = offsets[n]
            offsets[n] += 4
            a = batches[n][offset:offset+4].cuda()
            pa = feature_batches[n][offset:offset+4]
            data = inputs(model, 4, n, 16, torch.Generator(device='cuda').manual_seed(100000*epoch+4*batch_index),
                'cuda', pa, training=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(a, a.abs().sum(-1), *data)
            loss, diagnostics = loss_function(a, out)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            records.append(dict(loss=loss.item(), cut=out['best'].mean().item(), grad_norm=float(norm), **diagnostics))
        score = run_ablation.validate(model, validation, val_parity, 'cuda')
        row = dict(epoch=epoch, sizes=sizes, validation=score, seconds=time.perf_counter()-start,
            **{k: float(np.mean([r[k] for r in records])) for k in records[0]})
        history.append(row)
        if score > best:
            best = score
            save('best.pt', epoch, score)
        save('last.pt', epoch, score)
        (args.output/'history.json').write_text(json.dumps(history, indent=2))
        print(json.dumps(row), flush=True)
