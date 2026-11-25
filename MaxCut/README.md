# Maximum Cut

Run from the release root. Training uses generated graphs; inference uses
the 35 bundled Gset graphs.

## To train

```bash
python -B -u -m MaxCut.train --output runs/maxcut_training
```

## Inference with trained checkpoints

```bash
python -B -u -m MaxCut.evaluate --checkpoint MaxCut/checkpoints/model/best.pt --m 1000 --output runs/maxcut_example.json
```

Main experiments: `bash MaxCut/reproduce.sh`. Results: `runs/maxcut/summary.json`.
