import torch

from DKS.structural import anchor_features
from DKS.model import cardinality_lmo, induced_edges


@torch.inference_mode()
def evaluate_queries(model, adjacency, degree, core, anchors, draws, k, chunk):
    if chunk < 1:
        raise ValueError('chunk must be positive')
    scores_all = []
    best_score = best_indices = None
    for left in range(0, anchors.shape[1], chunk):
        right = min(left + chunk, anchors.shape[1])
        features = anchor_features(adjacency, degree, core, anchors[:, left:right], k)
        queries = model.queries(features, draws[:, left:right])
        vertices = cardinality_lmo(queries, k)
        scores = induced_edges(adjacency, vertices)
        local_score, local_best = scores[0].max(dim=0)
        indices = queries[0, local_best].topk(k).indices
        if best_score is None:
            best_score, best_indices = local_score, indices
        else:
            better = local_score > best_score
            best_indices = torch.where(better, indices, best_indices)
            best_score = torch.maximum(best_score, local_score)
        scores_all.append(scores.flatten())
        del features, queries, vertices, scores, indices, local_score, local_best
    return torch.cat(scores_all), best_indices.sort().values
