import numpy as np
import torch
from MaxCut.ablation import inputs

@torch.inference_mode()
def validate(model, validation, parity, device):
    model.eval()
    scores = []
    for start in range(0, len(validation), 4):
        a = validation[start:start+4].to(device)
        pa = parity[start:start+4] if parity is not None else None
        data = inputs(model, len(a), a.shape[-1], 50,
            torch.Generator(device=device).manual_seed(990001+start), device, pa)
        out = model(a, a.abs().sum(-1), *data)
        scores.extend((out['best']/(a.abs().sum((-2, -1))/2)).cpu().tolist())
    return float(np.mean(scores))
