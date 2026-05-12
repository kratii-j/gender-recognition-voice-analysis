from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

FEATURE_NAMES = [
    "meanfreq",
    "sd",
    "median",
    "Q25",
    "Q75",
    "IQR",
    "skew",
    "kurt",
    "sp.ent",
    "sfm",
    "mode",
    "centroid",
    "meanfun",
    "minfun",
    "maxfun",
    "meandom",
    "mindom",
    "maxdom",
    "dfrange",
    "modindx",
]

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "artifacts" / "voice_gender_mlp.pt"


class GenderMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...] = (64, 32)):
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.15))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)
