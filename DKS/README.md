# Densest k-Subgraph

Run from the release root. Training uses generated graphs; inference uses
12 graphs from the [SNAP dataset collection](https://snap.stanford.edu/data/)
with $k=100$. Graph data is not included in this repository.

## Download test data

Download and prepare all 12 graphs once(run from root folder):

```bash
python -B -u -m DKS.data
```

The command creates `DKS/data/<dataset>/adjacency.npz` and `node_ids.npy`.
It treats edges as undirected, removes duplicate edges and self-loops, and
relabels nodes in ascending original-ID order. All nodes appearing in the
edge list are retained. Large graphs such as Orkut require substantial CPU
RAM and disk space during preprocessing.

To prepare selected graphs, use `python -B -u -m DKS.data --datasets Facebook ca-HepTh`.
Run preparation only for datasets whose output folders do not already exist.

Original edge-list downloads:

| Dataset          | Download                                                                                                   |
| ---------------- | ---------------------------------------------------------------------------------------------------------- |
| Facebook         | [facebook_combined.txt.gz](https://snap.stanford.edu/data/facebook_combined.txt.gz)                         |
| web-Stanford     | [web-Stanford.txt.gz](https://snap.stanford.edu/data/web-Stanford.txt.gz)                                   |
| com-Youtube      | [com-youtube.ungraph.txt.gz](https://snap.stanford.edu/data/bigdata/communities/com-youtube.ungraph.txt.gz) |
| soc-Pokec        | [soc-pokec-relationships.txt.gz](https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz)             |
| as-Skitter       | [as-skitter.txt.gz](https://snap.stanford.edu/data/as-skitter.txt.gz)                                       |
| com-Orkut        | [com-orkut.ungraph.txt.gz](https://snap.stanford.edu/data/bigdata/communities/com-orkut.ungraph.txt.gz)     |
| ca-HepTh         | [ca-HepTh.txt.gz](https://snap.stanford.edu/data/ca-HepTh.txt.gz)                                           |
| ca-CondMat       | [ca-CondMat.txt.gz](https://snap.stanford.edu/data/ca-CondMat.txt.gz)                                       |
| com-DBLP         | [com-dblp.ungraph.txt.gz](https://snap.stanford.edu/data/bigdata/communities/com-dblp.ungraph.txt.gz)       |
| roadNet-CA       | [roadNet-CA.txt.gz](https://snap.stanford.edu/data/roadNet-CA.txt.gz)                                       |
| wiki-Talk        | [wiki-Talk.txt.gz](https://snap.stanford.edu/data/wiki-Talk.txt.gz)                                         |
| soc-LiveJournal1 | [soc-LiveJournal1.txt.gz](https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz)                           |

## To train

```bash
python -B -u -m DKS.train --n 500 --m 32 --epochs 30 --output runs/dks_training
```

## Inference with trained checkpoints

```bash
python -B -u -m DKS.evaluate --checkpoint DKS/checkpoints/model/best.pt --k 100 --m 1000 --chunk 16 --seed 42 --output runs/dks_example
```

Main experiments: `bash DKS/reproduce.sh`. Results: `runs/dks/summary.json`.
