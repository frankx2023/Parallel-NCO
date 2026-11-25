# Maximum Coverage

Run from the release root. Data includes Random500, Random1000, Random5000,
Random10000, Rail and Twitch.

## To train

```bash
python -B -u -m MC.train --config MC/checkpoints/random1000_k20_m50/config.json --data MC/data/uniform/max_covering_1000_train.pkl --k 20 --m 50 --epochs 50 --output runs/mc_training
```

## Inference with trained checkpoints

```bash
bash MC/reproduce.sh
```

Runs all eight main settings: Random500 ($k=10,50$), Random1000
($k=20,100$), Rail ($k=20,50$), and Twitch ($k=20,50$).
Each completed setting prints its mean coverage and time and saves a separate
result JSON under `runs/mc/`. The combined results are saved in
`runs/mc/summary.json`. For another run, use `bash MC/reproduce.sh runs/mc_new`.
