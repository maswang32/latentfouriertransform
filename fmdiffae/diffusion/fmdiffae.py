import torch
import torch.nn as nn
from tqdm import tqdm
from fmdiffae.arc.fftmask import FFTMask


class FMDiffAE(nn.Module):
    def __init__(
        self,
        decoder,
        encoder,
        datashape=(80, 256),
        sigma_data=0.5,
        use_tanh=False,
        use_mask_as_condition=False,
        **fftmask_kwargs,
    ):
        """
        Frequency-Masked Diffusion AutoEncoder.
        The Decoder is an EDM-Style Diffusion Model.

        Attributes:
            decoder (nn.Module): diffusion network
            encoder (nn.Module): maps an input condition to a feature map
            datashape (Tuple): shape of the data example, excluding batch size.
            sigma_data (float): per-dim standard deviation of the dataset
        """
        super().__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.sigma_data = sigma_data
        self.datashape = datashape
        assert len(datashape) > 0

        self.use_tanh = use_tanh
        self.use_mask_as_condition = use_mask_as_condition
        self.fftmask = FFTMask(**fftmask_kwargs)

    def forward(self, y, P_mean=-1.2, P_std=1.2):
        batch_size = y.shape[0]

        # Feature Map
        if not self.use_tanh:
            z = self.encoder(y)
        else:
            z = torch.tanh(self.encoder(y))

        if self.use_mask_as_condition:
            z_masked, fft_mask = self.fftmask(z, return_mask=True)
        else:
            z_masked = self.fftmask(z)
            fft_mask = None

        # Noisy Data
        sigmas = torch.exp(P_mean + torch.randn(batch_size, device=y.device) * P_std)
        sigmas = self._add_dims(sigmas, batch_size)
        c_skip, c_out, c_in, c_noise = self._get_cs(sigmas)
        n = torch.randn_like(y, device=y.device) * sigmas
        noisy = c_in * (y + n)

        # Decoder Output
        decoder_in = torch.cat((noisy, z_masked), dim=1)
        decoder_out = self.decoder(decoder_in, c_noise, cond=fft_mask)

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
        return_intermediates=False,
    ):
        device, dtype = inputs.device, inputs.dtype
        batch_size = inputs.shape[0]

        # Feature Map
        if not self.use_tanh:
            z = self.encoder(inputs)
        else:
            z = torch.tanh(self.encoder(inputs))

        if self.use_mask_as_condition:
            z_masked, fft_mask = self.fftmask(
                z, return_mask=True, lows=lows, highs=highs
            )
        else:
            z_masked = self.fftmask(z, lows=lows, highs=highs)
            fft_mask = None

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

        if return_intermediates:
            intermediates = []

        for step in tqdm(range(num_steps), desc="Generating", leave=False):
            sigma = sigmas[step]
            sigma_next = sigmas[step + 1]
            delta_sigma = sigma_next - sigma

            if return_intermediates:
                intermediates.append(x_curr.cpu())

            d_curr = self._get_derivative(
                x_curr.expand(batch_size, *x_curr.shape[1:]),
                self._add_dims(sigma, batch_size),
                z_masked=z_masked,
                fft_mask=fft_mask,
            )

            x_next = x_curr + d_curr * delta_sigma

            if step != num_steps - 1:
                d_next = self._get_derivative(
                    x_next.expand(batch_size, *x_next.shape[1:]),
                    self._add_dims(sigma_next, batch_size),
                    z_masked=z_masked,
                    fft_mask=fft_mask,
                )
                d = (d_curr + d_next) / 2

                x_next = x_curr + d * delta_sigma

            x_curr = x_next

        if return_intermediates:
            intermediates.append(x_curr.cpu())
            return x_curr.cpu(), torch.stack(intermediates, dim=0), sigmas.cpu()

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

    def _denoise(self, x, sigma, z_masked, fft_mask):
        c_skip, c_out, c_in, c_noise = self._get_cs(sigma)
        net_in = torch.cat((c_in * x, z_masked), dim=1)
        return c_skip * x + c_out * self.decoder(net_in, c_noise, cond=fft_mask)

    def _get_derivative(self, x, sigma, z_masked, fft_mask):
        return (
            x - self._denoise(x, sigma=sigma, z_masked=z_masked, fft_mask=fft_mask)
        ) / sigma
