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
        fft_mask=None,
        cfg_scale=1.0,
        blend_weights=None,
        init_noise=None,
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
        if (inputs is None) == (zs is None):
            raise ValueError("Exactly one of `inputs` or `zs` must be provided")

        if (lows is None) != (highs is None):
            raise ValueError("Both `lows` and `highs` must be provided together")

        if lows is not None and fft_mask is not None:
            raise ValueError("Cannot pass both `fft_mask` and `lows`/`highs`ss")

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
                blend_weights = torch.as_tensor(
                    blend_weights, dtype=dtype, device=device
                )
            else:
                blend_weights = blend_weights.to(dtype=dtype, device=device)

            num_to_blend = blend_weights.shape[-1]
            blend_weights = blend_weights / torch.sum(
                blend_weights, dim=-1, keepdim=True
            )

        # Apply Frequency Masking to zs
        if lows is not None or fft_mask is not None:
            zs = zs.view(-1, *z_shape)

            if lows is not None:
                if not isinstance(lows, torch.Tensor):
                    lows = torch.as_tensor(lows, dtype=dtype, device=device)
                    highs = torch.as_tensor(highs, dtype=dtype, device=device)
                else:
                    lows = lows.to(dtype=dtype, device=device)
                    highs = highs.to(dtype=dtype, device=device)

                lows = lows.view(-1)
                highs = highs.view(-1)

            if fft_mask is not None:
                if not isinstance(fft_mask, torch.Tensor):
                    fft_mask = torch.as_tensor(fft_mask, device=device)
                else:
                    fft_mask = fft_mask.to(device=device)

                fft_mask = fft_mask.view(-1, fft_mask.shape[-1])

            zs = self.freq_mask(
                zs,
                lows=lows,
                highs=highs,
                fft_mask=fft_mask,
            )

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
        if init_noise is None:
            x_curr = (
                torch.randn((batch_size, *self.datashape), dtype=dtype, device=device)
                * sigma_max
            )
        else:
            x_curr = init_noise.to(dtype=dtype, device=device)

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

    @torch.no_grad()
    def batch_generate(
        self,
        batch_size,
        device,
        inputs=None,
        zs=None,
        lows=None,
        highs=None,
        fft_mask=None,
        cfg_scale=1.0,
        blend_weights=None,
        init_noise=None,
        num_steps=35,
        sigma_max=80,
        sigma_min=0.002,
        rho=7,
        outer_pbar=True,
        inner_pbar=False,
    ):
        # Compute total number of examples to generate
        if inputs is not None:
            total = inputs.shape[0]
            ndim = inputs.ndim
        elif zs is not None:
            total = zs.shape[0]
            ndim = zs.ndim
        else:
            raise ValueError("Inputs or zs must be provided to generate")

        if blend_weights is not None:
            if ndim < len(self.datashape) + 2:
                raise ValueError(
                    "Batch dim must be provided to batch_generate with blending: "
                    "inputs or zs shape must be (total, num_to_blend, ...)"
                )

            if not isinstance(blend_weights, torch.Tensor):
                blend_weights = torch.as_tensor(blend_weights)

            # generate fcn allows broadcasting blend_weights across batch dim.
            # In this case, expanding is needed to ensure proper slicing.
            if blend_weights.ndim == 1:
                blend_weights = blend_weights.expand(total, -1)

        # Manage batching
        indices = torch.arange(total)

        def opt_slice(x, idx):
            if x is not None:
                x = x[idx]
                if isinstance(x, torch.Tensor):
                    x = x.to(device)
            return x

        all_outs = []

        iterator = indices.split(batch_size)
        if outer_pbar:
            iterator = tqdm(iterator, desc="Generating Batches", leave=False)

        for batch_indices in iterator:
            output = self.generate(
                inputs=opt_slice(inputs, batch_indices),
                zs=opt_slice(zs, batch_indices),
                lows=opt_slice(lows, batch_indices),
                highs=opt_slice(highs, batch_indices),
                fft_mask=opt_slice(fft_mask, batch_indices),
                cfg_scale=cfg_scale,
                blend_weights=opt_slice(blend_weights, batch_indices),
                init_noise=opt_slice(init_noise, batch_indices),
                num_steps=num_steps,
                sigma_max=sigma_max,
                sigma_min=sigma_min,
                rho=rho,
                pbar=inner_pbar,
            )
            all_outs.append(output.cpu())

        return torch.cat(all_outs, dim=0)

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
