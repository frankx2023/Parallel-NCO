import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data


def sigmoid_scaling_layer(x: torch.Tensor, k: float, sharpness: float = 1.0) -> torch.Tensor:
    dtype = x.dtype
    y = torch.sigmoid(x * sharpness)
    n = y.numel()
    m = y.mean()
    y_min = torch.min(y)
    y_max = torch.max(y)
    u = k / n
    scale_lower = (m - y_min) / u
    scale_upper = (y_max - m) / (1 - u)
    s = torch.max(scale_lower, scale_upper)
    _ = (y - m) / s + u
    return y.to(dtype)


def sigmoid_only_layer(x: torch.Tensor, sharpness: float = 1.0) -> torch.Tensor:
    return torch.sigmoid(x * sharpness)


class BipartiteData(Data):
    def __init__(self, edge_index, x_src, x_dst):
        super().__init__()
        self.edge_index = edge_index
        self.x1 = x_src
        self.x2 = x_dst

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.x1.size(0)], [self.x2.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


def build_graph_from_weights_sets(weights, sets, device=torch.device("cpu")):
    max_red_degree = max([len(s) for s in sets]) if sets else 1
    max_weight = max(weights) if weights else 1

    red_degrees = torch.tensor([len(s) for s in sets], dtype=torch.float, device=device).unsqueeze(-1)
    bias = torch.ones_like(red_degrees)
    normalized_red_degrees = red_degrees / max_red_degree

    neighbor_weight_means = []
    neighbor_weight_maxes = []
    for s in sets:
        if s:
            neighbor_weights = [weights[idx] for idx in s]
            neighbor_weight_means.append(sum(neighbor_weights) / len(neighbor_weights))
            neighbor_weight_maxes.append(max(neighbor_weights))
        else:
            neighbor_weight_means.append(0)
            neighbor_weight_maxes.append(0)

    weight_means = torch.tensor(neighbor_weight_means, dtype=torch.float, device=device).unsqueeze(-1)
    weight_maxes = torch.tensor(neighbor_weight_maxes, dtype=torch.float, device=device).unsqueeze(-1)

    x_src = torch.cat(
        [
            bias,
            red_degrees,
            normalized_red_degrees,
            weight_means / max_weight,
            weight_maxes / max_weight,
        ],
        dim=-1,
    )

    blue_degrees = [0] * len(weights)
    blue_red_degree_sums = [0] * len(weights)
    for red_set in sets:
        red_degree = len(red_set)
        for idx in red_set:
            blue_degrees[idx] += 1
            blue_red_degree_sums[idx] += red_degree

    max_blue_degree = max(blue_degrees) if blue_degrees else 1
    weight_tensor = torch.tensor(weights, dtype=torch.float, device=device).unsqueeze(-1)
    blue_deg_tensor = torch.tensor(blue_degrees, dtype=torch.float, device=device).unsqueeze(-1)
    norm_weight_tensor = weight_tensor / max_weight
    norm_blue_deg = blue_deg_tensor / max_blue_degree

    avg_red_degrees = [
        blue_red_degree_sums[b_idx] / blue_degrees[b_idx] if blue_degrees[b_idx] else 0
        for b_idx in range(len(weights))
    ]
    avg_red_deg_tensor = torch.tensor(avg_red_degrees, dtype=torch.float, device=device).unsqueeze(-1)

    x_dst = torch.cat(
        [
            weight_tensor,
            blue_deg_tensor,
            norm_weight_tensor,
            norm_blue_deg,
            avg_red_deg_tensor / max_red_degree,
        ],
        dim=-1,
    )

    idx1, idx2 = [], []
    for r, bl in enumerate(sets):
        for b in bl:
            idx1.append(r)
            idx2.append(b)
    edge_index = torch.tensor([idx1, idx2], dtype=torch.long, device=device)

    return BipartiteData(edge_index, x_src, x_dst)


class ConvLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.w2s = SAGEConv((hidden_size, hidden_size), hidden_size)
        self.s2w = SAGEConv((hidden_size, hidden_size), hidden_size)
        self.bn_w2s = nn.BatchNorm1d(hidden_size)
        self.bn_s2w = nn.BatchNorm1d(hidden_size)
        self.ln = nn.LayerNorm(hidden_size)


class GNN(torch.nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        sharpness_init: float = 1.0,
        hidden_size: int = 64,
        scaling_mode: str = "sigmoid_only",
    ):
        super().__init__()
        self.dropout = dropout
        self.sharpness = sharpness_init
        self.hidden_size = hidden_size
        self.scaling_mode = scaling_mode

        self.gconv1_w2s = SAGEConv((5, 5), hidden_size)
        self.gconv1_s2w = SAGEConv((5, 5), hidden_size)
        self.bn1_w2s = nn.BatchNorm1d(hidden_size)
        self.bn1_s2w = nn.BatchNorm1d(hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)

        self.conv_layers = nn.ModuleList()
        for _ in range(3):
            self.conv_layers.append(ConvLayer(hidden_size))

        self.fc1 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, 1)

    def _apply_conv_layer(self, x1, x2, layer, g_edge_index, rev_edge):
        s_to_w = layer.w2s((x2, x1), rev_edge)
        w_to_s = layer.s2w((x1, x2), g_edge_index)

        s_to_w = layer.bn_w2s(s_to_w)
        w_to_s = layer.bn_s2w(w_to_s)

        s_to_w = F.leaky_relu(s_to_w, negative_slope=0.2)
        w_to_s = F.leaky_relu(w_to_s, negative_slope=0.2)

        x1_new = s_to_w + x1
        x2_new = w_to_s + x2

        x1_new = F.dropout(x1_new, p=self.dropout, training=self.training)
        x1_new = layer.ln(x1_new)

        return x1_new, x2_new

    def _red_embeddings(self, g):
        rev_edge = torch.stack((g.edge_index[1], g.edge_index[0]), dim=0)

        s_to_w = self.gconv1_w2s((g.x2, g.x1), rev_edge)
        w_to_s = self.gconv1_s2w((g.x1, g.x2), g.edge_index)
        s_to_w = self.bn1_w2s(s_to_w)
        w_to_s = self.bn1_s2w(w_to_s)
        s_to_w = F.leaky_relu(s_to_w, negative_slope=0.2)
        w_to_s = F.leaky_relu(w_to_s, negative_slope=0.2)
        x1, x2 = s_to_w, w_to_s
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x1 = self.ln1(x1)

        for layer in self.conv_layers:
            x1, x2 = self._apply_conv_layer(x1, x2, layer, g.edge_index, rev_edge)

        x1_centered = x1 - x1.mean(dim=0, keepdim=True)
        x1_scaled = x1_centered * torch.sqrt(torch.tensor(self.hidden_size, device=x1.device))

        return x1_scaled

    def forward(self, g, f, batch=None, noise_std: float = 0.0):
        def normalize_logits(x):
            x_mean = x.mean()
            x_std = x.std() + 1e-6
            return (x - x_mean) / x_std

        x1_scaled = self._red_embeddings(g)

        x = self.fc1(x1_scaled)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x).squeeze(-1)

        if self.training and noise_std > 0:
            x = x + torch.randn_like(x) * noise_std

        self.last_raw = x.detach()

        unique_batch = torch.unique(batch)
        result = torch.zeros_like(x)

        for i, b_idx in enumerate(unique_batch):
            mask = batch == b_idx
            graph_k = f[i] if f.dim() > 0 and i < f.size(0) else f
            x_norm = normalize_logits(x[mask])
            if self.scaling_mode == "sigmoid_only":
                x_scaled = sigmoid_only_layer(x_norm, sharpness=self.sharpness)
            elif self.scaling_mode == "sigmoid_scaling":
                x_scaled = sigmoid_scaling_layer(x_norm, graph_k, sharpness=self.sharpness)
            elif self.scaling_mode == "none":
                x_scaled = x_norm
            else:
                raise ValueError(f"Unknown scaling_mode: {self.scaling_mode}")
            result[mask] = x_scaled

        return result


class GaussianDirectionGNN(GNN):

    def __init__(
        self,
        num_directions: int,
        initial_log_variance: float,
        dropout: float = 0.0,
        hidden_size: int = 64,
    ):
        if num_directions < 2:
            raise ValueError("num_directions must be at least 2")

        super().__init__(
            dropout=dropout,
            sharpness_init=1.0,
            hidden_size=hidden_size,
            scaling_mode="none",
        )
        self.num_directions = num_directions
        del self.fc2
        self.mean_head = nn.Linear(hidden_size + 1, num_directions)
        self.log_variance_head = nn.Linear(hidden_size + 1, num_directions)

        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_variance_head.weight)
        nn.init.constant_(self.log_variance_head.bias, initial_log_variance)

    def forward(self, g, center):
        red_embeddings = self._red_embeddings(g)
        hidden = self.fc1(red_embeddings)
        hidden = F.leaky_relu(hidden, negative_slope=0.2)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        conditioned = torch.cat([hidden, center.unsqueeze(-1)], dim=-1)
        mean_offsets = self.mean_head(conditioned)
        log_variances = self.log_variance_head(conditioned)

        if not torch.isfinite(mean_offsets).all():
            raise RuntimeError("Gaussian direction means contain non-finite values")
        if not torch.isfinite(log_variances).all():
            raise RuntimeError("Gaussian direction log-variances contain non-finite values")

        return mean_offsets, log_variances


class ParallelGaussianModel(nn.Module):

    def __init__(
        self,
        num_directions: int,
        initial_log_variance: float,
        dropout: float,
        sharpness_init: float,
        hidden_size: int,
        scaling_mode: str,
    ):
        super().__init__()
        self.num_directions = num_directions
        self.center_network = GNN(
            dropout=dropout,
            sharpness_init=sharpness_init,
            hidden_size=hidden_size,
            scaling_mode=scaling_mode,
        )
        self.direction_network = GaussianDirectionGNN(
            num_directions=num_directions,
            initial_log_variance=initial_log_variance,
            dropout=dropout,
            hidden_size=hidden_size,
        )

    @property
    def sharpness(self):
        return self.center_network.sharpness

    @sharpness.setter
    def sharpness(self, value):
        self.center_network.sharpness = value

    def forward(self, g, budgets, batch, noise_std: float = 0.0):
        center = self.center_network(
            g,
            budgets,
            batch=batch,
            noise_std=noise_std,
        )
        mean_offsets, log_variances = self.direction_network(g, center)
        return center, mean_offsets, log_variances


