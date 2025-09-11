import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from lightning.pytorch.callbacks import Callback

from fmdiffae.utils.fad import get_embeddings_vggish, compute_fad_from_embeddings
from fmdiffae.lightning.callbacks import print_once


class EDM(nn.Module):
    def __init__(
        self,
        net,
        datashape,
        sigma_data=0.5,
    ):
        super().__init__()
        self.net = net
        self.datashape = datashape
        self.sigma_data = sigma_data

    def forward(self, y, P_mean=-1.2, P_std=1.2):
        batch_size = y.shape[0]
        sigmas = torch.exp(P_mean + torch.randn(batch_size, device=y.device) * P_std)
        sigmas = self._add_dims(sigmas, N=batch_size)
        c_skip, c_out, c_in, c_noise = self._get_cs(sigmas)
        n = torch.randn_like(y) * sigmas
        net_out = self.net(c_in * (y + n), c_noise)
        target = (y - c_skip * (y + n)) / c_out
        loss = nn.functional.mse_loss(net_out, target)
        return loss

    @torch.no_grad()
    def generate(
        self,
        batch_size=1,
        init_noise=None,
        num_steps=35,
        sigma_max=80,
        sigma_min=0.002,
        rho=7,
        heun=True,
        pbar=False,
        guidance_fcn=None,
        guidance_scale=1.0,
        guidance_mode="x0",
        ilvr_mode=None,
        ilvr_lows=None,
        ilvr_highs=None,
        ilvr_reference=None,
        ilvr_nfft=None,
        **guidance_fcn_kwargs,
    ):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

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

            d_curr = self._get_derivative(
                x_curr,
                self._add_dims(sigma, batch_size),
                guidance_fcn=guidance_fcn,
                guidance_scale=guidance_scale,
                guidance_mode=guidance_mode,
                **guidance_fcn_kwargs,
            )

            x_next = x_curr + d_curr * delta_sigma

            if heun and step != num_steps - 1:
                # Heun Correction
                d_next = self._get_derivative(
                    x_next,
                    self._add_dims(sigma_next, batch_size),
                    guidance_fcn=guidance_fcn,
                    guidance_scale=guidance_scale,
                    guidance_mode=guidance_mode,
                    **guidance_fcn_kwargs,
                )
                d = (d_curr + d_next) / 2
                x_next = x_curr + d * delta_sigma

            x_curr = x_next

        if ilvr_reference is not None:
            if ilvr_mode == "cond":
                callback_fcn = ilvr_callback
            elif ilvr_mode == "blend":
                callback_fcn = dual_ilvr_callback
            else:
                raise ValueError

            x_curr = callback_fcn(
                x_curr,
                ilvr_lows,
                ilvr_highs,
                ilvr_reference,
                sigma=self._add_dims(sigma_next, batch_size),
                n_fft=ilvr_nfft,
            )

        return x_curr

    def _denoise(self, x, sigma):
        c_skip, c_out, c_in, c_noise = self._get_cs(sigma)
        return c_skip * x + c_out * self.net(c_in * x, c_noise)

    def _get_derivative(
        self,
        x,
        sigma,
        guidance_fcn=None,
        guidance_scale=1.0,
        guidance_mode="x0",
        **guidance_fcn_kwargs,
    ):
        denoised = self._denoise(x, sigma=sigma)
        d = (x - denoised) / sigma

        if guidance_fcn is not None and guidance_scale > 0:
            if guidance_mode == "x0":
                pred = denoised
            elif guidance_mode == "xt":
                pred = x

            with torch.enable_grad():
                pred = pred.detach().requires_grad_(True)
                loss = guidance_fcn(pred, **guidance_fcn_kwargs)
                g = torch.autograd.grad(
                    loss, pred, create_graph=False, retain_graph=False
                )[0].detach()

            # derivative points to the forward process
            # (delta_sigma is negative during generation)
            d = d + guidance_scale * g

        return d

    def _add_dims(self, x, N):
        assert x.ndim < 2
        if x.ndim == 0 or x.shape[0] != N:
            x = x.expand(N)
        return x.view((N,) + (1,) * len(self.datashape))

    def _get_cs(self, sigma: torch.Tensor) -> torch.Tensor:
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = (self.sigma_data * sigma) / torch.sqrt(self.sigma_data**2 + sigma**2)
        c_in = torch.rsqrt(sigma**2 + self.sigma_data**2)
        c_noise = 0.25 * torch.log(sigma.clamp_min(1e-12))
        c_noise = c_noise.reshape(-1)
        return c_skip, c_out, c_in, c_noise


class FAD(Callback):
    def __init__(self, num_samples, num_steps, pbar=True):
        super().__init__()
        self.num_samples = num_samples
        self.num_steps = num_steps
        self.pbar = pbar

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        print_once("Computing FAD")

        # Defining Variables
        rank = pl_module.global_rank
        world_size = trainer.world_size
        sample_rate = trainer.datamodule.sample_rate
        pbar = self.pbar and rank == 0

        print("rank", rank)
        print("world size", trainer.world_size)

        assert self.num_samples % world_size == 0, (
            "World size must divide number of total samples"
        )
        examples_per_rank = self.num_samples // world_size

        # Using EMA
        model = (
            pl_module.ema_model.module
            if getattr(pl_module, "ema_model", None)
            else pl_module.model
        )

        # Generate Sample
        samples = model.generate(
            batch_size=examples_per_rank,
            num_steps=self.num_steps,
            pbar=pbar,
        )

        # Invert
        audios = pl_module.transform.batched_inverse_transform(samples, pbar=pbar)
        audios = audios - torch.mean(audios, dim=-1, keepdim=True)
        audios = audios / audios.abs().amax(-1, keepdim=True).clamp_min(1e-8)

        # Compute Embeddings
        embs = get_embeddings_vggish(
            audios.cpu(),
            fs=sample_rate,
            pbar=pbar,
        )

        # Gather Embeddings
        if world_size > 1:
            embs = pl_module.all_gather(embs).cpu().flatten(0, 1)
            print(f"embs shape:{embs.shape}")

        # Compute FAD, Log
        # (N, T, E) -> (N*T, E)
        embs = embs.reshape(-1, embs.shape[-1]).numpy()
        ref_mean = np.load(trainer.datamodule.hparams.ref_mean_path)
        ref_cov = np.load(trainer.datamodule.hparams.ref_cov_path)
        fad = compute_fad_from_embeddings(
            mean1=ref_mean, cov1=ref_cov, embeddings2=embs
        )
        pl_module.log("FAD/max_fad", fad, sync_dist=True)


def spectral_guidance(
    x,
    guidance_lows,
    guidance_highs,
    w_iso,
    reference,
    w_reference,
    n_fft=1024,
):
    F = n_fft // 2 + 1
    v = torch.linspace(0, 1, F)

    # Select spectrum inside selected band
    fft_mask = (v >= guidance_lows.unsqueeze(1)) & (v <= guidance_highs.unsqueeze(1))
    fft_mask = fft_mask.unsqueeze(-2).to(device=x.device, dtype=x.dtype)

    # Flip mask for outside selected band
    inv_fft_mask = 1 - fft_mask

    x_spectrum = torch.fft.rfft(x, n=n_fft)

    # Isolation Loss
    x_power_spectrum = torch.abs(x_spectrum) ** 2
    loss_iso = torch.sum(inv_fft_mask * x_power_spectrum)

    # Reference Loss
    if reference is not None:
        ref_spectrum = torch.fft.rfft(reference, n=n_fft)
        squared_errors = torch.abs(ref_spectrum - x_spectrum) ** 2
        loss_reference = torch.sum(fft_mask * squared_errors)
    else:
        loss_reference = 0

    return w_iso * loss_iso + w_reference * loss_reference


def dual_spectral_guidance(
    x,
    both_guidance_lows,
    both_guidance_highs,
    references,
    n_fft=1024,
):
    loss = spectral_guidance(
        x,
        both_guidance_lows[0],
        both_guidance_highs[0],
        0,
        references[0],
        1,
        n_fft=n_fft,
    )
    loss += spectral_guidance(
        x,
        both_guidance_lows[1],
        both_guidance_highs[1],
        0,
        references[1],
        1,
        n_fft=n_fft,
    )
    return loss


def ilvr_callback(x, ilvr_lows, ilvr_highs, ilvr_reference, sigma, n_fft=1024):
    # Add Noise to Reference
    n = torch.randn_like(ilvr_reference) * sigma
    noisy_reference = ilvr_reference + n

    # Create FFT Mask
    F = n_fft // 2 + 1
    v = torch.linspace(0, 1, F)
    fft_mask = (v >= ilvr_lows.unsqueeze(1)) & (v <= ilvr_highs.unsqueeze(1))
    fft_mask = fft_mask.unsqueeze(-2).to(device=x.device, dtype=x.dtype)

    # Bandpass x_t and noisy reference
    x_bp = torch.fft.irfft(
        fft_mask * torch.fft.rfft(x.float(), n=n_fft),
    ).to(x.dtype)[..., : x.shape[-1]]

    noisy_reference_bp = torch.fft.irfft(
        fft_mask * torch.fft.rfft(noisy_reference.float(), n=n_fft),
    ).to(x.dtype)[..., : x.shape[-1]]

    return x - x_bp + noisy_reference_bp


def dual_ilvr_callback(
    x, both_ilvr_lows, both_ilvr_highs, references, sigma, n_fft=1024
):
    x = ilvr_callback(
        x,
        both_ilvr_lows[0],
        both_ilvr_highs[0],
        references[0],
        sigma=sigma,
        n_fft=n_fft,
    )
    x = ilvr_callback(
        x,
        both_ilvr_lows[1],
        both_ilvr_highs[1],
        references[1],
        sigma=sigma,
        n_fft=n_fft,
    )
    return x
