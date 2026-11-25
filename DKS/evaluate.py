import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import torch
from DKS.data import ROOT, DATA, SOURCES, file_hash
from DKS.inference import load_ours, query_inputs
from DKS.query_chunks import evaluate_queries
from DKS.evaluation_utils import cpu_score


def worker(args):
    folder = DATA/args.dataset
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    adjacency, checkpoint, model, a, d, c = load_ours(args.dataset, args.checkpoint)
    seed = args.seed+list(SOURCES).index(args.dataset)*10000+args.k
    torch.cuda.synchronize()
    preparation = time.perf_counter()-start
    start = time.perf_counter()
    with torch.inference_mode():
        anchors, draws = query_inputs(d,c,args.m,seed)
        scores, indices = evaluate_queries(model,a,d,c,anchors,draws,args.k,args.chunk)
        score = float(scores.max())
        selected = indices.cpu().numpy()
        torch.cuda.synchronize()
    inference = time.perf_counter()-start
    assert cpu_score(adjacency,selected,args.k)==score
    row = dict(dataset=args.dataset,n=adjacency.shape[0],k=args.k,m=args.m,chunk=args.chunk,
        edges=int(score),density=2*score/(args.k*(args.k-1)),
        preparation_s=preparation,inference_s=inference,time_s=preparation+inference,
        seed=seed,selected_nodes=selected.tolist(),
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    (args.output/(args.dataset+'.json')).write_text(json.dumps(row,indent=2)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='selected_nodes'}),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--datasets',nargs='+',choices=list(SOURCES),default=list(SOURCES))
    p.add_argument('--dataset',choices=list(SOURCES),help=argparse.SUPPRESS)
    p.add_argument('--k',type=int,default=100)
    p.add_argument('--m',type=int,default=200)
    p.add_argument('--chunk',type=int,default=16)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.dataset is not None:
        worker(args)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for dataset in args.datasets:
        subprocess.run([sys.executable,'-m','DKS.evaluate','--checkpoint',str(args.checkpoint.resolve()),
            '--dataset',dataset,'--k',str(args.k),'--m',str(args.m),'--chunk',str(args.chunk),
            '--seed',str(args.seed),
            '--output',str(args.output.resolve())],cwd=ROOT,check=True)
        rows.append(json.loads((args.output/(dataset+'.json')).read_text()))
        (args.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':
    main()
