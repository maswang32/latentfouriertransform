import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from latentft.arc.unet1d import GroupNorm


class PointwiseNet(nn.Module):
    def __init__(
        self,
        in_channels=80,
        out_channels=16,
        hidden_channels=[512, 512, 512, 512, 512, 512, 512, 512],
    ):
        super().__init__()
        self.hidden_channels = hidden_channels
        if len(self.hidden_channels) > 0:
            self.in_conv = nn.Conv1d(in_channels, hidden_channels[0], 1)
            self.residual_blocks = nn.ModuleList(
                [
                    ResidualBlock(hidden_channels[i], hidden_channels[i + 1])
                    for i in range(len(hidden_channels) - 1)
                ]
            )
            self.out_conv = nn.Conv1d(hidden_channels[-1], out_channels, 1)
        else:
            self.out_conv = nn.Conv1d(in_channels, out_channels, 1)

        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"Pointwise Net Number of Parameters: {self.num_params:,}")

    def forward(self, x):
        if len(self.hidden_channels) > 0:
            x = self.in_conv(x)
            for block in self.residual_blocks:
                x = block(x)
        return self.out_conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm0 = GroupNorm(in_channels)
        self.conv0 = nn.Conv1d(in_channels, out_channels, 1)

        self.norm1 = GroupNorm(out_channels)
        self.conv1 = nn.Conv1d(out_channels, out_channels, 1)

        with torch.no_grad():
            self.conv1.weight.mul_(1e-5)

        if in_channels != out_channels:
            self.skip_proj = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x):
        residual_stream = x
        x = self.conv0(F.silu(self.norm0(x)))
        x = self.conv1(F.silu(self.norm1(x)))
        x = (x + self.skip_proj(residual_stream)) / math.sqrt(2)
        return x
