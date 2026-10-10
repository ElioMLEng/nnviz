"""Example of an importable model file for nnviz.

Required:  build() -> nn.Module
Optional:  example_input() -> tensor   (otherwise give the input spec in the UI)
           CLASSES = [...]             (labels for the input selector / outputs)
"""
import torch
import torch.nn as nn


class SmallResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU())
        self.block = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                                   nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16))
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(16, 5)

    def forward(self, x):
        x = self.stem(x)
        x = self.act(x + self.block(x))     # residual connection
        return self.head(self.pool(x).flatten(1))


def build():
    return SmallResNet().eval()


def example_input():
    return torch.randn(1, 3, 32, 32)


CLASSES = ["example", "noise"]
