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
        sigmas = self._add_dims(sigmas, N=batch_size)
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
        inputs=None,
        zs=None,
        lows=None,
        highs=None,
        cfg_scale=1.0,
        blend_weights=None,
        num_steps=35,
        sigma_max=80,
        sigma_min=0.002,
        rho=7,
        pbar=False,
    ):
        """
        Assume data and latents (ignoring batch-like dims) have the same number of dims.
        Note: if self.use_tanh is true, and zs are passed instead of inputs,
            tanh must be applied before passing zs.
        """
        # Require inputs xor latents as condition
        assert (inputs is not None) ^ (zs is not None)

        # Both lows and highs must be specified, or neither
        assert (lows is not None) == (highs is not None)

        # Compute zs if necessary, and flatten/squeeze them
        if zs is None:
            zs = self.encoder(inputs.view(-1, *self.datashape))
            if self.use_tanh:
                zs = torch.tanh(zs)

        # Get shape, datatype, and device of zs
        z_shape = zs.shape[-len(self.datashape) :]
        dtype, device = zs.dtype, zs.device

        # Compute blend weights
        if blend_weights is None:
            num_to_blend = 1
        else:
            if not isinstance(blend_weights, torch.Tensor):
                blend_weights = torch.tensor(blend_weights, dtype=dtype, device=device)
            else:
                blend_weights = blend_weights.to(dtype=dtype, device=device)
            num_to_blend = blend_weights.shape[-1]

        # Apply Frequency Masking to zs
        if lows is not None:
            zs = zs.view(-1, *z_shape)

            if not isinstance(lows, torch.Tensor):
                lows = torch.tensor(lows, dtype=dtype, device=device)
                highs = torch.tensor(highs, dtype=dtype, device=device)
            else:
                lows = lows.to(dtype=dtype, device=device)
                highs = highs.to(dtype=dtype, device=device)

            zs = self.freq_mask(zs, lows=lows.view(-1), highs=highs.view(-1))

        # Full Shape
        zs = zs.view(-1, num_to_blend, *z_shape)
        batch_size = zs.shape[0]

        # Broadcast blend weights across the batch dimension, if per-trajectory
        # blend weights are not provided
        if blend_weights is not None and blend_weights.ndim == 1:
            blend_weights = blend_weights.expand(batch_size, -1)

        if cfg_scale != 1.0:
            # Append null condition
            zs = torch.cat(
                (
                    zs,
                    torch.zeros((zs.shape[0], 1, *z_shape), dtype=dtype, device=device),
                ),
                dim=1,
            )

            # If blend_weights are not provided, create them to assist with CFG
            if blend_weights is None:
                blend_weights = torch.ones(
                    (batch_size, num_to_blend), dtype=dtype, device=device
                )

            # Adjust blend weights to include null condition
            blend_weights = torch.cat(
                (
                    blend_weights * cfg_scale,
                    torch.full(
                        (batch_size, 1), 1 - cfg_scale, dtype=dtype, device=device
                    ),
                ),
                dim=1,
            )

        # Initialize generation and noise levels
        x_curr = (
            torch.randn((batch_size, *self.datashape), dtype=dtype, device=device)
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
        sigmas = torch.cat((sigmas, torch.zeros(1, dtype=dtype, device=device)))

        # Generation Loop
        iterator = range(num_steps)
        if pbar:
            iterator = tqdm(iterator, desc="Generating", leave=False)
        for step in iterator:
            sigma = sigmas[step]
            sigma_next = sigmas[step + 1]
            delta_sigma = sigma_next - sigma

            d_curr = self._get_combined_derivative(
                x=x_curr,
                sigma=sigma,
                zs=zs,
                blend_weights=blend_weights,
            )

            x_next = x_curr + d_curr * delta_sigma

            if step != num_steps - 1:
                # Huen Correction
                d_next = self._get_combined_derivative(
                    x=x_next,
                    sigma=sigma_next,
                    zs=zs,
                    blend_weights=blend_weights,
                )
                d = (d_curr + d_next) / 2
                x_next = x_curr + d * delta_sigma

            x_curr = x_next

        return x_curr

    def _denoise(self, x, sigma, z):
        c_skip, c_out, c_in, c_noise = self._get_cs(sigma)
        net_in = torch.cat((c_in * x, z), dim=1)
        return c_skip * x + c_out * self.decoder(net_in, c_noise)

    def _get_derivative(self, x, sigma, z):
        denoised = self._denoise(x, sigma=sigma, z=z)
        return (x - denoised) / sigma

    def _get_combined_derivative(self, x, sigma, zs, blend_weights):
        batch_size, num_to_blend = zs.shape[0], zs.shape[1]

        ds = self._get_derivative(
            x.unsqueeze(1).expand(batch_size, num_to_blend, *x.shape[1:]).flatten(0, 1),
            sigma=self._add_dims(sigma, N=batch_size * num_to_blend),
            z=zs.flatten(0, 1),
        )

        if blend_weights is not None:
            ds = ds.unflatten(0, (batch_size, num_to_blend))
            ds = torch.sum(
                ds
                * blend_weights.view(
                    (batch_size, num_to_blend) + (1,) * len(self.datashape)
                ),
                dim=1,
            )

        return ds

    def _add_dims(self, x, N):
        assert x.ndim < 2
        if x.ndim == 0 or x.shape[0] != N:
            x = x.expand(N)
        return x.view((N,) + (1,) * len(self.datashape))

    def _get_cs(self, sigma: torch.Tensor) -> torch.Tensor:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = (self.sigma_data * sigma) / torch.sqrt(self.sigma_data**2 + sigma**2)
        c_in = torch.sqrt(sigma**2 + self.sigma_data**2).reciprocal()
        c_noise = 0.25 * torch.log(sigma.clamp_min(1e-12))
        c_noise = c_noise.reshape(-1)
        return c_skip, c_out, c_in, c_noise
