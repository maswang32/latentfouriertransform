import torch
import torch.nn as nn
from tqdm import tqdm


class FMDiffAE(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        freq_mask,
        datashape,
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
        self.freq_mask = freq_mask
        self.datashape = datashape
        self.sigma_data = sigma_data
        self.use_tanh = use_tanh

    def forward(self, y, P_mean=-1.2, P_std=1.2):
        batch_size = y.shape[0]

        # Get Feature Map
        z = self.encoder(y)
        if self.use_tanh:
            z = torch.tanh(z)

        # Apply Frequency Mask
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
        cfg_scale=1.0,
        blend_weights=None,
        num_steps=35,
        sigma_max=80,
        sigma_min=0.002,
        rho=7,
        pbar=False,
    ):
        device, dtype = inputs.device, inputs.dtype
        num_inputs = inputs.shape[0]
        batch_size = num_inputs if blend_weights is None else 1

        # Feature Map
        assert lows.shape[0] == num_inputs
        assert highs.shape[0] == num_inputs

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
                x_curr,
                self._add_dims(sigma, batch_size),
                z=z,
                cfg_scale=cfg_scale,
                blend_weights=blend_weights,
            )

            x_next = x_curr + d_curr * delta_sigma

            if step != num_steps - 1:
                d_next = self._get_derivative(
                    x_next,
                    self._add_dims(sigma_next, batch_size),
                    z=z,
                    cfg_scale=cfg_scale,
                    blend_weights=blend_weights,
                )
                d = (d_curr + d_next) / 2

                x_next = x_curr + d * delta_sigma

            x_curr = x_next

        return x_curr

    def _add_dims(self, x, batch_size):
        assert x.ndim < 2
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

    def _get_derivative(self, x, sigma, z, cfg_scale=1.0, blend_weights=None):
        batch_size = x.shape[0]
        num_conditions = z.shape[0]
        assert sigma.shape[0] == batch_size

        if blend_weights is None:
            assert num_conditions == batch_size
        else:
            assert blend_weights.shape[0] == num_conditions
            assert batch_size == 1

        if cfg_scale == 1.0:
            # N - batch size for the denoiser.
            N = num_conditions
        else:
            N = num_conditions + batch_size
            z = torch.cat((z, torch.zeros((batch_size, *z.shape[1:]), device=z.device)))

        # R - number of times to duplicate x and sigma
        R = int(N / batch_size)

        x_expanded = x.unsqueeze(0).expand(R, *x.shape).reshape(-1, *x.shape[1:])
        sigma_expanded = (
            sigma.unsqueeze(0).expand(R, *sigma.shape).reshape(-1, *sigma.shape[1:])
        )

        denoised_expanded = self._denoise(x_expanded, sigma=sigma_expanded, z=z)
        denoised_cond = denoised_expanded[:num_conditions]

        if blend_weights is not None:
            denoised_cond = torch.sum(
                denoised_cond * self._add_dims(blend_weights, blend_weights.shape[0]),
                dim=0,
                keepdim=True,
            )

        if cfg_scale != 1.0:
            denoised_uncond = denoised_expanded[-batch_size:]
            denoised = denoised_uncond + cfg_scale * (denoised_cond - denoised_uncond)
        else:
            denoised = denoised_cond

        return (x - denoised) / sigma
