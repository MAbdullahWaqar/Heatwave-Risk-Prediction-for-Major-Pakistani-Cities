"""GRU + Attention risk classifier (`RNNAttentionClassifier`; must match notebook arch)."""

from __future__ import annotations

import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class RNNAttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_cities: int,
        num_classes: int,
        rnn_type: str = "lstm",
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
        embed_dim: int = 8,
    ):
        super().__init__()
        self.city_emb = nn.Embedding(num_cities, embed_dim)
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = AttentionPool(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, city_idx: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        ctx = self.attn(out)
        city_vec = self.city_emb(city_idx)
        return self.head(torch.cat([ctx, city_vec], dim=1))


def load_lstm_checkpoint(path, device: torch.device | None = None) -> tuple[RNNAttentionClassifier, dict]:
    if device is None:
        device = torch.device("cpu")
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = payload["config"]
    feature_cols: list = payload["feature_cols"]
    city_to_idx: dict = payload["city_to_idx"]
    n_feat = len(feature_cols)
    num_cities = len(city_to_idx)
    n_cls = 4
    for k, t in payload["model_state_dict"].items():
        if "head.3.weight" in k and hasattr(t, "shape"):
            n_cls = int(t.shape[0])
            break
    _mn = str(payload.get("model_name", "")).upper()
    if "GRU" in _mn:
        _rnn = "gru"
    else:
        _rnn = "lstm"  # LSTM_Attn or any non-GRU RNNAttention name

    m = RNNAttentionClassifier(
        input_dim=n_feat,
        num_cities=num_cities,
        num_classes=n_cls,
        rnn_type=_rnn,
        hidden_dim=cfg.get("hidden_dim", 64),
        num_layers=cfg.get("num_layers", 2),
        dropout=cfg.get("dropout", 0.25),
        embed_dim=cfg.get("embed_dim", 8),
    )
    m.load_state_dict(payload["model_state_dict"])
    m.to(device)
    m.eval()
    return m, payload
