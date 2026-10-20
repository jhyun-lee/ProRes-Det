"""A small, plain convolutional residual predictor for a restoration baseline."""

from dataclasses import asdict, dataclass, fields

import torch.nn as nn

PLAIN_CNN_ARCH_NAME = "plain_cnn"


@dataclass
class PlainCNNConfig:
    width: int = 32
    depth: int = 5

    def __post_init__(self):
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 1:
            raise ValueError("plain_cnn.width must be a positive integer")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 2:
            raise ValueError("plain_cnn.depth must be an integer >= 2 (total convolutions)")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        known = {field.name for field in fields(cls)}
        unknown = set(values or {}) - known
        if unknown:
            raise ValueError(f"unknown plain_cnn settings: {', '.join(sorted(unknown))}")
        return cls(**(values or {}))

    def is_default(self):
        return self == PlainCNNConfig()

    def tag(self):
        return f"CNN-W{self.width}-D{self.depth}"

    def describe(self):
        return (f"plain CNN: {self.depth} convolutions, width={self.width}, "
                f"receptive field={2 * self.depth + 1}x{2 * self.depth + 1}; "
                "6-channel input -> 3-channel tanh residual")


class PlainCNN(nn.Module):
    """Conv3x3 + ReLU hidden layers, Conv3x3 + Tanh output; no resampling."""

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or PlainCNNConfig()
        width = self.cfg.width
        layers = [nn.Conv2d(6, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(self.cfg.depth - 2):
            layers.extend([nn.Conv2d(width, width, 3, padding=1), nn.ReLU(inplace=True)])
        layers.extend([nn.Conv2d(width, 3, 3, padding=1), nn.Tanh()])
        self.body = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.body(inputs)
