# Code Repository for paper "Fast Neural Combinatorial Optimization via Parallel Oracles"

Code and trained checkpoints for MC, QAP, MaxCut and DKS.
MC, QAP and MaxCut data are included. DKS graphs can be downloaded from SNAP using the commands in [DKS/README.md](DKS/README.md).

## Setup

Use a CUDA-enabled PyTorch installation in a Conda environment:

```bash
conda create -n parallel_nco python=3.12 -y
conda activate parallel_nco
pip install -r requirements.txt
```

Run the commands in the problem READMEs from this directory to retrain and reproduce results with provided checkpoints.

## Reproduce

After preparing the DKS data, reproduce the main experiments using the supplied checkpoints:

```bash
bash reproduce.sh
```

Run one problem using `bash MC/reproduce.sh`, `bash QAP/reproduce.sh`,
`bash MaxCut/reproduce.sh` or `bash DKS/reproduce.sh`.
Results and logs are saved under `runs/`, with aggregate results in `summary.json`.
Pass a new output folder as the first script argument for another run.
