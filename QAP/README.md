# Quadratic Assignment

Run from the release root. Training uses QAP32; inference uses QAPLIB.

## To train

```bash
python -B -u -m QAP.train --config QAP/checkpoints/m5000_base/config.json --m 5000 --output runs/qap_training
```

## Inference with trained checkpoints

```bash
python -B -u -m QAP.evaluate_table --checkpoint QAP/checkpoints/m5000/selected.pt --samples 4 --output runs/qap_example.json
```

Main experiments: `bash QAP/reproduce.sh`. Results: `runs/qap/summary.json`.
