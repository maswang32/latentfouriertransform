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
        num_steps=35,
        sigma_max=80,
        sigma_min=0.002,
        rho=7,
        pbar=False,
    ):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

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

            d_curr = self._get_derivative(
                x_curr,
                self._add_dims(sigma, batch_size),
            )

            x_next = x_curr + d_curr * delta_sigma

            if step != num_steps - 1:
                # Huen Correction
                d_next = self._get_derivative(
                    x_next,
                    self._add_dims(sigma_next, batch_size),
                )
                d = (d_curr + d_next) / 2
                x_next = x_curr + d * delta_sigma

            x_curr = x_next

        return x_curr

    def _denoise(self, x, sigma):
        c_skip, c_out, c_in, c_noise = self._get_cs(sigma)
        return c_skip * x + c_out * self.net(c_in * x, c_noise)

    def _get_derivative(self, x, sigma):
        return (x - self._denoise(x, sigma=sigma)) / sigma

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
