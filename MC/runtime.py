import pickle
import random
import numpy as np
import torch
from MC.models import ParallelGaussianModel, build_graph_from_weights_sets
from MC.queries import (maximum_coverage_costs, parallel_cardinality_lmo,
    sample_gaussian_directions, sparse_maximum_coverage_costs)


def load_model(path, device):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = make_model(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.sharpness = checkpoint['sharpness']
    return model, checkpoint['config']

def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def make_model(cfg):
    method = 'queries'
    common = dict(dropout=cfg["model"]["dropout"],
                  sharpness_init=cfg["training"]["sharpness"],
                  hidden_size=cfg["model"]["hidden_size"],
                  scaling_mode=cfg["training"]["scaling_mode"],
                  num_directions=cfg[method]["num_directions"])
    return ParallelGaussianModel(
        initial_log_variance=cfg[method]["initial_log_variance"], **common)


def read_data(path):
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    records = []
    for name, weights, sets in raw:
        graph = build_graph_from_weights_sets(weights, sets, device="cpu")
        graph.num_nodes = len(sets) + len(weights)
        graph.coverage_weights = torch.tensor(weights, dtype=torch.float32)
        if len(sets) <= 1000 and len(weights) <= 2000:
            graph.coverage_incidence = torch.zeros(len(sets), len(weights))
            graph.coverage_incidence[graph.edge_index[0], graph.edge_index[1]] = 1.
        records.append((str(name), graph, weights, sets))
    return records


@torch.inference_mode()
def predict(model, graph, k, budget, batch):
    center, means, logvars = model(graph, budget, batch, noise_std=0.)
    directions, _, _ = sample_gaussian_directions(center, means, logvars)
    vertices = parallel_cardinality_lmo(directions, k)
    if 'coverage_incidence' in graph:
        costs = maximum_coverage_costs(vertices, graph.coverage_incidence, graph.coverage_weights)
    else:
        costs = sparse_maximum_coverage_costs(vertices, graph.edge_index, graph.coverage_weights)
    index = costs.argmin()
    return vertices[index].nonzero().flatten(), costs[index], vertices
