import math
import torch
import torch.nn as nn


class FFTMask(nn.Module):
    def __init__(self, length=256):
        super().__init__()

        assert math.log2(length).is_integer(), "length must be a power of 2"

        self.length = length
        self.F = length // 2 + 1  # rfft length
        self.G = int(math.log2(length))

        # m represents fft bins in each group
        m = torch.zeros(self.G, self.F)
        splits = 2 ** torch.arange(0, self.G)
        for i in range(self.G):
            if i == 0:
                m[i, 0:1] = 1
            elif i == 1:
                m[i, 1:3] = 1
            else:
                m[i, splits[i] // 2 + 1 : splits[i] + 1] = 1

        self.register_buffer("m", m)

        # Assign normalized frequencies to bins
        self.register_buffer("c", torch.linspace(0, 1, self.F).unsqueeze(0))

    def forward(self, x, lows=None, highs=None, return_mask=False):
        assert x.ndim == 3, "x must have 3 dimensions"
        assert (lows is None) == (highs is None)
        assert x.shape[-1] == self.length, "input length must match FFT Length"

        batch_size = x.shape[0]
        device, dtype = x.device, x.dtype

        if lows is None:
            thresholds = torch.rand(batch_size, 1, device=device)
            scores = torch.rand(batch_size, self.G, device=device)
            fft_mask = (scores > thresholds).to(dtype) @ self.m.to(dtype)
        else:
            c = self.c.to(device=device, dtype=dtype)
            fft_mask = ((c >= lows.unsqueeze(1)) & (c <= highs.unsqueeze(1))).to(dtype)

        x = torch.fft.irfft(
            fft_mask.unsqueeze(1) * torch.fft.rfft(x), n=self.length, dim=-1
        )

        if not return_mask:
            return x
        else:
            return x, fft_mask
