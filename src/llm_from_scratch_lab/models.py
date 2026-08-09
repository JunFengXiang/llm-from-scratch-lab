"""Small models used before the repository reaches the GPT chapters."""

from __future__ import annotations

from torch import nn


class ToyMLP(nn.Module):
    """A minimal classifier that keeps attention on the training loop."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32, num_classes: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs):
        return self.network(inputs)
