import math
import torch
import torch.nn as nn


class CorrelatedFFTMask(nn.Module):
    def __init__(self, n_fft=1024, sigma=0.5, p=2, logscale=True, eps=1e-6):
        super().__init__()
        assert math.log2(n_fft).is_integer(), "n_fft must be a power of 2"

        self.n_fft = n_fft
        self.F = n_fft // 2 + 1  # rfft length

        self.register_buffer(
            "v", torch.linspace(0, 1, self.F), persistent=False
        )  # Normalized Frequencies

        # Transformation used correlate fft bin scores
        if sigma > 0:
            if logscale:
                self.log_v = torch.log(self.v + eps)  # Normalized freqs in log space
                axis = self.log_v
            else:
                axis = self.v

            k = torch.exp(
                -0.5 * ((torch.abs(axis[:, None] - axis[None, :]) / sigma) ** p)
            )
            k = k / torch.sqrt(torch.sum(k**2, axis=0, keepdim=True))
        else:
            k = torch.eye(self.F)

        self.register_buffer("k", k, persistent=False)

    def forward(self, x, lows=None, highs=None, fft_mask=None):
        assert x.ndim == 3, "input to fftmask must have 3 dimensions"
        assert (lows is None) == (highs is None)

        batch_size, length = x.shape[0], x.shape[-1]
        device, dtype = x.device, x.dtype

        if fft_mask is None:
            if lows is None:
                scores = (
                    torch.randn(batch_size, self.F, device=device, dtype=dtype) @ self.k
                )
                thresholds = torch.randn(batch_size, 1, device=device, dtype=dtype)
                fft_mask = scores > thresholds
            else:
                fft_mask = (self.v >= lows.unsqueeze(1)) & (
                    self.v <= highs.unsqueeze(1)
                )

        return torch.fft.irfft(
            fft_mask.to(dtype).unsqueeze(1) * torch.fft.rfft(x.float(), n=self.n_fft),
        ).to(dtype)[..., :length]
