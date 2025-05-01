import torch
import torch.nn as nn
from tqdm import tqdm


class FMDiffAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        freq_mask,
        datashape=[80, 256],
        sigma_data=0.5,
        use_tanh=False,
    ):
        """
        Frequency-Masked Diffusion AutoEncoder.
        The Decoder is an EDM-Style Diffusion Model.

        Attributes:
            encoder (nn.Module): maps an input condition to a feature map
            decoder (nn.Module): diffusion network
            datashape (List[int]): shape of the data example, excluding batch size.
            sigma_data (float): per-dim standard deviation of the dataset
            use_tanh (bool): if we should apply tanh after the encoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sigma_data = sigma_data
        self.datashape = datashape
        assert len(datashape) > 0

        self.use_tanh = use_tanh
        self.freq_mask = freq_mask

    def forward(self, y, P_mean=-1.2, P_std=1.2):
        batch_size = y.shape[0]

        # Feature Map
        z = self.encoder(y)
        if self.use_tanh:
            z = torch.tanh(z)

        z = self.freq_mask(z)

        # Noisy Data
        sigmas = torch.exp(P_mean + torch.randn(batch_size, device=y.device) * P_std)
        sigmas = self._add_dims(sigmas, batch_size)
        c_skip, c_out, c_in, c_noise = self._get_cs(sigmas)
        n = torch.randn_like(y, device=y.device) * sigmas
        noisy = c_in * (y + n)

        # Decoder Output
        decoder_in = torch.cat((noisy, z), dim=1)
        decoder_out = self.decoder(decoder_in, c_noise)

        target = (y - c_skip * (y + n)) / c_out
        loss = nn.functional.mse_loss(decoder_out, target)
        return loss

    @torch.no_grad()
    def generate(
        self,
        inputs,
        lows,
        highs,
        num_steps=35,
        sigma_max=80,
        sigma_min=0.002,
        rho=7,
        pbar=False,
    ):
        device, dtype = inputs.device, inputs.dtype
        batch_size = inputs.shape[0]

        # Feature Map
        z = self.encoder(inputs)
        if self.use_tanh:
            z = torch.tanh(z)

        z = self.freq_mask(z, lows=lows, highs=highs)

        x_curr = (
            torch.randn((batch_size, *self.datashape), device=device, dtype=dtype)
            * sigma_max
        )

        sigmas = (
            torch.linspace(
                sigma_max ** (1 / rho),
                sigma_min ** (1 / rho),
                num_steps,
                dtype=dtype,
                device=device,
            )
            ** rho
        )
        sigmas = torch.cat((sigmas, torch.zeros(1, device=device, dtype=dtype)))

        iterator = range(num_steps)
        if pbar:
            iterator = tqdm(iterator, desc="Generating", leave=False)

        for step in iterator:
            sigma = sigmas[step]
            sigma_next = sigmas[step + 1]
            delta_sigma = sigma_next - sigma

            d_curr = self._get_derivative(
                x_curr.expand(batch_size, *x_curr.shape[1:]),
                self._add_dims(sigma, batch_size),
                z=z,
            )

            x_next = x_curr + d_curr * delta_sigma

            if step != num_steps - 1:
                d_next = self._get_derivative(
                    x_next.expand(batch_size, *x_next.shape[1:]),
                    self._add_dims(sigma_next, batch_size),
                    z=z,
                )
                d = (d_curr + d_next) / 2

                x_next = x_curr + d * delta_sigma

            x_curr = x_next

        return x_curr

    def _add_dims(self, x, batch_size=1):
        if x.ndim == 0 or x.shape[0] != batch_size:
            x = x.expand(batch_size)
        return x.view((batch_size,) + (1,) * len(self.datashape))

    def _get_cs(self, sigma: torch.Tensor) -> torch.Tensor:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = (self.sigma_data * sigma) / torch.sqrt(self.sigma_data**2 + sigma**2)
        c_in = torch.sqrt(sigma**2 + self.sigma_data**2).reciprocal()
        c_noise = 0.25 * torch.log(sigma.clamp_min(1e-12))
        c_noise = c_noise.reshape(-1)
        return c_skip, c_out, c_in, c_noise

    def _denoise(self, x, sigma, z):
        c_skip, c_out, c_in, c_noise = self._get_cs(sigma)
        net_in = torch.cat((c_in * x, z), dim=1)
        return c_skip * x + c_out * self.decoder(net_in, c_noise)

    def _get_derivative(self, x, sigma, z):
        return (x - self._denoise(x, sigma=sigma, z=z)) / sigma
