import torch
import torch.nn as nn
from .config import MODEL_HIDDEN_SIZES


def _build_mlp(input_dim, hidden_sizes, output_dim, dropout):
    layers = []
    last_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(last_dim, size))
        layers.append(nn.LayerNorm(size))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        last_dim = size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim,
        actions_per_source,
        max_sources,
        hidden_sizes=MODEL_HIDDEN_SIZES,
        dropout=0.1,
    ):
        super().__init__()
        self.actions_per_source = actions_per_source
        self.max_sources = max_sources
        self.body = _build_mlp(obs_dim, hidden_sizes, hidden_sizes[-1], dropout)
        self.policy_head = nn.Linear(hidden_sizes[-1], max_sources * actions_per_source)
        self.value_head = nn.Linear(hidden_sizes[-1], 1)

    def forward(self, obs):
        features = self.body(obs)
        logits = self.policy_head(features)
        logits = logits.view(-1, self.max_sources, self.actions_per_source)
        value = self.value_head(features)
        return logits, value
