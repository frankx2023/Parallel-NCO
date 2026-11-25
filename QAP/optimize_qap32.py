import argparse
import copy
import json
import time
from pathlib import Path

import torch

from QAP import core as base
from QAP.eval_parallel_qaplib import load_model
from QAP.parallel_qap import parallel_qap_training_loss, predict_parallel_qap

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "checkpoints/m5000_base/best.pt"
RUN = ROOT / "runs/m5000_selected"


def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def extend_features(model, args, max_n):
    for features, seed in [(model.featA, args["feat_seed"]), (model.featB, args["feat_seed"]+999)]:
        gen = torch.Generator().manual_seed(seed)
        random_features = torch.randn(max_n, args["rproj"], generator=gen)
        features.R = torch.cat((features.R, random_features[features.max_n:].to(features.R.device)), dim=0)
        features.max_n = max_n
    args["max_n"] = max_n


@torch.no_grad()
def validation(model, A, B, samples, num_gaussians=5000):
    model.eval()
    base.set_seed(9100)
    costs = []
    for i in range(A.shape[0]):
        p = predict_parallel_qap(model, A[i:i+1], B[i:i+1],
                                 num_gaussians=num_gaussians, samples_per_gaussian=samples)
        costs.append(p.best_cost.item())
    return sum(costs)/len(costs)


def main(source=SOURCE, run=RUN, num_gaussians=5000):
    run.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    base.set_seed(2026)
    model, args = load_model(str(source), torch.device("cuda"))
    assert args["n"] == 32 and args["num_gaussians"] == num_gaussians
    args = copy.deepcopy(args)
    A, B = base.load_flow_pos_data(str(ROOT/"data/erdos32_0.3_F_train.npy"),
                                 str(ROOT/"data/erdos32_0.3_positions_train.npy"), torch.device("cuda"))
    Va, Vb = base.load_flow_pos_data(str(ROOT/"data/erdos32_0.3_F_test.npy"),
                                   str(ROOT/"data/erdos32_0.3_positions_test.npy"), torch.device("cuda"))
    with torch.no_grad():
        before = model(Va[:1], Vb[:1])
        extend_features(model, args, 256)
        after = model(Va[:1], Vb[:1])
        for x,y in zip(before,after): torch.testing.assert_close(x,y,rtol=0,atol=0)
    del before, after
    initial_state = cpu_state(model)
    initial_cost = validation(model, Va[:64], Vb[:64], 1, num_gaussians)
    history = [{"stage": "initial", "validation64_cost": initial_cost}]
    best_cost, best_state, best_source = initial_cost, initial_state, "initial"
    print("[INITIAL] validation64", initial_cost, flush=True)

    def fit(stage, lr, count, state):
        nonlocal best_cost, best_state, best_source
        model.load_state_dict(state, strict=True)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        local_best, local_state = float("inf"), None
        for epoch in range(2):
            base.set_seed(2026+epoch)
            order = torch.randperm(count, device="cuda")
            model.train()
            start = time.perf_counter()
            for offset in range(0, count, 8):
                index = order[offset:offset+8]
                opt.zero_grad(set_to_none=True)
                result = parallel_qap_training_loss(model, A[index], B[index], num_gaussians=num_gaussians,
                    samples_per_gaussian=1, kkt_ridge=1e-2, kkt_solver="woodbury")
                result.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
            cost = validation(model, Va[:64], Vb[:64], 1, num_gaussians)
            record = {"stage":stage, "lr":lr, "epoch":epoch, "training_instances":count,
                      "validation64_cost":cost, "seconds":time.perf_counter()-start}
            history.append(record)
            print("[TUNE]", record, flush=True)
            if cost < local_best:
                local_best, local_state = cost, cpu_state(model)
            if cost < best_cost:
                best_cost, best_state, best_source = cost, cpu_state(model), f"{stage}_epoch{epoch}"
        return local_best, local_state

    pilots = []
    for lr in [1e-6, 3e-6]:
        cost,state = fit(f"pilot_lr{lr}", lr, 512, initial_state)
        pilots.append((cost, lr, state))
    _, lr, _ = min(pilots, key=lambda x:x[0])
    fit("full", lr, A.shape[0], best_state)
    model.load_state_dict(best_state, strict=True)
    sampling = [{"samples_per_gaussian":s, "validation64_cost":validation(model,Va[:64],Vb[:64],s,num_gaussians)}
                for s in [1,4]]
    chosen_samples = min(sampling,key=lambda x:x["validation64_cost"])["samples_per_gaussian"]
    args.update(num_gaussians=num_gaussians, samples_per_gaussian=chosen_samples, lr=lr)
    report = {"fine_tune_history":history, "selected_state":best_source,
              "selected_validation64_cost":best_cost, "sampling_search":sampling,
              "selected_samples_per_gaussian":chosen_samples,
              "validation256_cost":validation(model,Va,Vb,chosen_samples,num_gaussians)}
    torch.save({"model_state":cpu_state(model), "args":args, "epoch":-1,
                "validation_cost":report["validation256_cost"], "optimization":report}, run/"selected.pt")
    with (run/"optimization.json").open("w") as f: json.dump(report,f,indent=2)
    print("[SELECTED]",report,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=RUN)
    parser.add_argument('--m', type=int, default=5000)
    cli = parser.parse_args()
    main(cli.source, cli.output, cli.m)
