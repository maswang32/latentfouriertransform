import numpy as np
import matplotlib.pyplot as plt
import torch
import wandb

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from fmdiffae.utils.fad import get_embeddings_vggish, compute_fad_from_embeddings


@rank_zero_only
def print_once(msg):
    print(msg)


class PlotFeatureMap(Callback):
    def __init__(self, valid_idx):
        super().__init__()
        self.valid_idx = valid_idx

    @rank_zero_only
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        model = (
            pl_module.ema_model.module
            if getattr(pl_module, "ema_model", None)
            else pl_module.model
        )

        inputs = trainer.datamodule.valid_ds[self.valid_idx]
        inputs = inputs.to(pl_module.device).unsqueeze(0)

        feature_map = model.encoder(inputs).cpu().numpy()[0]
        feature_map = feature_map[np.argsort(np.mean(feature_map, axis=-1))]

        # Plot Feature Map
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        num_features = feature_map.shape[0]
        num_frames = feature_map.shape[1]

        im = axs[0].imshow(feature_map, aspect=num_frames / (3 * num_features))
        axs[0].set_title("Feature Map, Sorted by Mean Value")
        fig.colorbar(im, ax=axs[0])

        # Plot Spectrum
        spectrum = np.fft.rfft(feature_map, axis=1)
        magnitude = np.abs(spectrum) / np.max(np.abs(spectrum))
        log_mag = 20 * np.log10(magnitude)

        for feature_fft in log_mag:
            axs[1].plot(
                np.linspace(0, 1, feature_fft.shape[-1]),
                feature_fft,
                alpha=0.3,
                color="black",
            )
        axs[1].set_title("Feature Spectrums")
        axs[1].set_xlabel("Normalized Frequency")
        axs[1].set_ylabel("Magnitude (dB)")

        trainer.logger.log_image(
            key="feature_maps", images=[fig], step=trainer.global_step
        )
        plt.close(fig)


class GenerateExamples(Callback):
    def __init__(self, valid_idx, num_steps, low_highs, pbar=False):
        super().__init__()
        self.valid_idx = valid_idx
        self.low_highs = torch.tensor(low_highs, dtype=torch.float32)
        self.num_examples = len(self.low_highs)
        self.num_steps = num_steps
        self.pbar = pbar

    @rank_zero_only
    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        device = pl_module.device
        inputs = trainer.datamodule.valid_ds[self.valid_idx].to(device)
        inputs = inputs.expand(self.num_examples, *inputs.shape).contiguous()

        lows, highs = self.low_highs.to(device).unbind(dim=-1)

        model = (
            pl_module.ema_model.module
            if getattr(pl_module, "ema_model", None)
            else pl_module.model
        )

        samples = model.generate(
            inputs=inputs,
            lows=lows,
            highs=highs,
            num_steps=self.num_steps,
            pbar=self.pbar,
        )

        audios = pl_module.transform.batched_inverse_transform(samples, pbar=self.pbar)
        audios = audios.cpu().numpy()
        samples = samples.clamp_(-1, 1).add_(1).div_(2).cpu().numpy()
        rows = []
        for (low, high), sample, audio in zip(self.low_highs, samples, audios):
            rows.append(
                [
                    f"{low:.3f}-{high:.3f}",
                    wandb.Image(sample, caption=f"{low:.3f}-{high:.3f}"),
                    wandb.Audio(audio, sample_rate=trainer.datamodule.sample_rate),
                ]
            )

        trainer.logger.log_table(
            key="examples/table",
            columns=["range", "spectrogram", "audio"],
            data=rows,
            step=trainer.global_step,
        )


class FADAndReconstruction(Callback):
    def __init__(self, num_samples, num_steps, low_highs, pbar=False):
        super().__init__()
        self.num_samples = num_samples
        self.low_highs = torch.tensor(low_highs, dtype=torch.float32)
        self.num_low_highs = len(self.low_highs)
        self.num_steps = num_steps
        self.pbar = pbar

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module):
        # Defining Variables
        device = pl_module.device
        rank = pl_module.global_rank
        world_size = trainer.world_size
        batch_size = trainer.datamodule.batch_size
        sample_rate = trainer.datamodule.sample_rate
        low_highs = self.low_highs.to(device)
        pbar = self.pbar and rank == 0

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

            batch_samples = model.generate(
                inputs=batch_inputs,
                lows=lows,
                highs=highs,
                num_steps=self.num_steps,
                pbar=pbar,
            )

            batch_mses = torch.nn.functional.mse_loss(
                batch_samples, batch_inputs, reduction="none"
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

        # Debugging
        torch.set_printoptions(threshold=float("inf"))
        print_once(indices)  # Debugging

        if rank == 0:
            # (HL*N, T, E) -> (HL, N*T, E)
            embs = embs.reshape(self.num_low_highs, -1, embs.shape[-1]).numpy()
            mses = mses.reshape(self.num_low_highs, self.num_samples).mean(-1).tolist()

            fads = [
                compute_fad_from_embeddings(
                    trainer.datamodule.valid_vggish_embeddings, x
                )
                for x in embs
            ]

            fad_names = [f"FAD/{low:.3f}-{high:.3f}" for (low, high) in self.low_highs]
            mse_names = [
                f"Recon_MSE/{low:.3f}-{high:.3f}" for (low, high) in self.low_highs
            ]

            metrics = {
                **{name: fad for name, fad in zip(fad_names, fads)},
                **{name: mse for name, mse in zip(mse_names, mses)},
            }
            trainer.logger.experiment.log(
                metrics,
                step=trainer.global_step,
            )
            pl_module.log("FAD/max_fad", max(fads), sync_dist=False)
