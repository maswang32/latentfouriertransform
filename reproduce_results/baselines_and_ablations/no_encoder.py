import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data.dataset import Dataset

from lightning.pytorch.callbacks import Callback

from fmdiffae.diffusion.fmdiffae import FMDiffAE
from fmdiffae.utils.fad import get_embeddings_vggish, compute_fad_from_embeddings
from fmdiffae.lightning.callbacks import print_once


class FMDiffAENoEncoder(FMDiffAE):
    def __init__(
        self,
        decoder,
        freq_mask,
        datashape,
        downsampling_factor,
        sigma_data=0.5,
    ):
        super().__init__(
            encoder=None,
            decoder=decoder,
            freq_mask=freq_mask,
            datashape=datashape,
            sigma_data=sigma_data,
            use_tanh=False,
        )

        self.downsampling_factor = downsampling_factor
        self.resampler = torchaudio.transforms.Resample(downsampling_factor, 1)

    def forward(self, batch, P_mean=-1.2, P_std=1.2):
        audio = batch[0]
        spec = batch[1]

        z = self.resampler(audio).unsqueeze(1)

        batch_size = z.shape[0]

        # Apply Frequency Mask
        z = self.freq_mask(z)

        # Noisy Data
        sigmas = torch.exp(P_mean + torch.randn(batch_size, device=spec.device) * P_std)
        sigmas = self._add_dims(sigmas, N=batch_size)
        c_skip, c_out, c_in, c_noise = self._get_cs(sigmas)
        n = torch.randn_like(spec, device=spec.device) * sigmas
        noisy = c_in * (spec + n)

        # Decoder Output
        decoder_in = torch.cat((noisy, z), dim=1)
        decoder_out = self.decoder(decoder_in, c_noise)

        target = (spec - c_skip * (spec + n)) / c_out
        loss = nn.functional.mse_loss(decoder_out, target)
        return loss


class AudioSpecTensorDataset(Dataset):
    def __init__(self, audio_path, spec_path):
        super().__init__()

        self.audio = torch.from_numpy(np.load(audio_path))
        self.spec = torch.from_numpy(np.load(spec_path))

    def __len__(self):
        return len(self.spec)

    def __getitem__(self, idx):
        return self.audio[idx], self.spec[idx]


class NoEncoderFADAndReconstruction(Callback):
    def __init__(self, num_samples, num_steps, low_highs, pbar=True):
        super().__init__()
        self.num_samples = num_samples
        self.low_highs = torch.tensor(low_highs, dtype=torch.float32)
        self.num_low_highs = len(self.low_highs)
        self.num_steps = num_steps
        self.pbar = pbar

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        print_once("Computing FAD")

        # Defining Variables
        device = pl_module.device
        rank = pl_module.global_rank
        world_size = trainer.world_size
        batch_size = trainer.datamodule.batch_size
        sample_rate = trainer.datamodule.sample_rate
        low_highs = self.low_highs.to(device)
        pbar = self.pbar and rank == 0

        print("rank", rank)
        print("world size", trainer.world_size)

        # Batching
        num_total_samples = self.num_samples * self.num_low_highs
        assert num_total_samples % batch_size == 0, (
            "Not Supported: Batch Sizes must divide num. total samples."
        )
        all_indices = torch.arange(num_total_samples).reshape(-1, batch_size)

        # Add padding so each rank sees the same number of batches
        pad = (-all_indices.shape[0]) % world_size
        all_indices = torch.nn.functional.pad(all_indices, (0, 0, 0, pad))

        assert all_indices.shape[0] % world_size == 0

        # Splitting across ranks
        rank_indices = all_indices.chunk(world_size)[rank]
        # Using EMA
        model = (
            pl_module.ema_model.module
            if getattr(pl_module, "ema_model", None)
            else pl_module.model
        )

        mses = []
        embs = []
        indices = []  # Debugging

        for batch_indices in rank_indices:
            batch_inputs = trainer.datamodule.valid_ds[
                batch_indices % self.num_samples
            ].to(device)
            lows, highs = low_highs[batch_indices // self.num_samples].unbind(dim=-1)

            batch_input_audios, batch_input_specs = batch_inputs

            zs = pl_module.model.resampler(batch_input_audios).unsqueeze(1)

            batch_samples = model.generate(
                zs=zs,
                lows=lows,
                highs=highs,
                num_steps=self.num_steps,
                pbar=pbar,
            )

            batch_mses = torch.nn.functional.mse_loss(
                batch_samples, batch_input_specs, reduction="none"
            ).mean(list(range(1, batch_samples.ndim)))

            audios = pl_module.transform.batched_inverse_transform(
                batch_samples, pbar=pbar
            )
            audios = audios - torch.mean(audios, dim=-1, keepdim=True)
            audios = audios / audios.abs().amax(-1, keepdim=True).clamp_min(1e-8)
            batch_embs = get_embeddings_vggish(
                audios.cpu(),
                fs=sample_rate,
                pbar=pbar,
            )

            embs.append(batch_embs)
            mses.append(batch_mses)
            indices.append(batch_indices)

        mses = torch.cat(mses, dim=0)
        embs = torch.cat(embs, dim=0)
        indices = torch.cat(indices, dim=0)

        if world_size > 1:
            mses = pl_module.all_gather(mses).flatten(0, 1)[:num_total_samples].cpu()

            # (W, (HL*N)/W, T, E) -> (HL*N, T, E)
            embs = pl_module.all_gather(embs).flatten(0, 1)[:num_total_samples].cpu()
            indices = pl_module.all_gather(indices)  # Debugging
        else:
            mses = mses.cpu()
            embs = embs.cpu()

        # Debugging - Ensure indices are gathered in correct order
        torch.set_printoptions(threshold=float("inf"))
        print_once(indices)

        # (HL*N, T, E) -> (HL, N*T, E)
        embs = embs.reshape(self.num_low_highs, -1, embs.shape[-1]).numpy()
        mses = mses.reshape(self.num_low_highs, self.num_samples).mean(-1).tolist()

        ref_mean = np.load(trainer.datamodule.hparams.ref_mean_path)
        ref_cov = np.load(trainer.datamodule.hparams.ref_cov_path)

        fads = [
            compute_fad_from_embeddings(mean1=ref_mean, cov1=ref_cov, embeddings2=x)
            for x in embs
        ]

        fad_names = [f"FAD/{low:.4f}-{high:.4f} Hz" for (low, high) in self.low_highs]
        mse_names = [
            f"Recon_MSE/{low:.4f}-{high:.4f} Hz" for (low, high) in self.low_highs
        ]

        metrics = {
            **{name: fad for name, fad in zip(fad_names, fads)},
            **{name: mse for name, mse in zip(mse_names, mses)},
        }
        trainer.logger.experiment.log(
            metrics,
            step=trainer.global_step,
        )
        pl_module.log("FAD/max_fad", max(fads), sync_dist=True)
