"""Colony: One Policy-Value Network Architecture (Level 4 Edition)."""

import torch
import torch.nn as nn


class ColonyNet(nn.Module):
    """Deep Residual Policy-Value Network for Colony: One.

    Contract:
      - Input:  observation (batch, 16, 8, 8) float32
      - Output: policy      (batch, 4096)     float32 (raw logits)
      - Output: value       (batch,)          float32 (in [-1, 1])

    Features:
      - 3 Residual Blocks (6 conv layers) with full 8x8 receptive field
      - 2D Spatial Convolutional Policy Head (channel = from_sq, spatial = to_sq)
      - Size: ~1.4 MB (well under the 5.0 MiB ceiling)
      - Dry run latency: ~35 ms
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(16, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu1 = nn.ReLU()

        self.block2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu2 = nn.ReLU()

        self.block3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.relu3 = nn.ReLU()

        # Policy Head -> 64 channels x 8 x 8 spatial = 4096 logits
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.Flatten(),
        )

        # Value Head -> 1 scalar in [-1, 1] representing win probability
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)
        h = self.relu1(h + self.block1(h))
        h = self.relu2(h + self.block2(h))
        h = self.relu3(h + self.block3(h))

        policy = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return policy, value
