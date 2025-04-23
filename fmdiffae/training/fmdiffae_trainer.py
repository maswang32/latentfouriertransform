import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Type, Dict, Any
from dataclasses import dataclass, field

from fmdiffae.training.trainer import Trainer, Config
from fmdiffae.diffusion.fmdiffae import FMDiffAE
from fmdiffae.arc.unet1d import UNet1d
from fmdiffae.transforms.bigvgan import BigVGANTransform
from fmdiffae.utils.fad import get_embeddings_vggish, compute_fad_from_embeddings
from fmdiffae.utils.evaluate import save_audio, save_image


@dataclass
class FMDiffAEConfig(Config):
    """
    Configuration Class for Frequency-Masked Diffusion AutoEncoder.
    Attributes:
        eval_low_highs: list of low and high cutoffs for generated examples
            during evaluation.
    """

    # Model Settings
    fmdiffae_kwargs: Dict[str, Any] = field(default_factory=dict)

    encoder_class: Type[torch.nn.Module] = UNet1d
    encoder_kwargs: Dict[str, Any] = field(default_factory=dict)

    decoder_class: Type[torch.nn.Module] = UNet1d
    decoder_kwargs: Dict[str, Any] = field(default_factory=dict)

    transform_class: type = BigVGANTransform
    transform_kwargs: Dict[str, Any] = field(default_factory=dict)

    num_steps_for_example: int = 100

    compute_fad: bool = True
    num_fad_examples: int = 256
    num_steps_for_fad: int = 35

    gen_low_highs: torch.Tensor = torch.tensor(
        [
            [0.0000, 0.0000],
            [0.0000, 0.0156],
            [0.0000, 0.0312],
            [0.0000, 0.0625],
            [0.0000, 0.1250],
            [0.0000, 0.2500],
            [0.0000, 0.5000],
            [0.0000, 1.0000],
            [0.0078, 0.0156],
            [0.0078, 0.0312],
            [0.0078, 0.0625],
            [0.0078, 0.1250],
            [0.0078, 0.2500],
            [0.0078, 0.5000],
            [0.0078, 1.0000],
            [0.0157, 0.0312],
            [0.0157, 0.0625],
            [0.0157, 0.1250],
            [0.0157, 0.2500],
            [0.0157, 0.5000],
            [0.0157, 1.0000],
            [0.0314, 0.0625],
            [0.0314, 0.1250],
            [0.0314, 0.2500],
            [0.0314, 0.5000],
            [0.0314, 1.0000],
            [0.0626, 0.1250],
            [0.0626, 0.2500],
            [0.0626, 0.5000],
            [0.0626, 1.0000],
            [0.1251, 0.2500],
            [0.1251, 0.5000],
            [0.1251, 1.0000],
            [0.2501, 0.5000],
            [0.2501, 1.0000],
            [0.5001, 1.0000],
        ]
    )

    eval_low_highs: torch.Tensor = torch.tensor(
        [
            [0.0000, 0.0000],
            [0.0078, 0.0156],
            [0.0157, 0.0312],
            [0.0314, 0.0625],
            [0.0626, 0.1250],
            [0.1251, 0.2500],
            [0.2501, 0.5000],
            [0.5001, 1.0000],
        ]
    )


class FMDiffAETrainer(Trainer):
    def __init__(self, config):
        self.config = config
        self.init_asserts()
        self.init_models()

        self.eval_low_highs = self.config.eval_low_highs.to(self.config.device)

        super().__init__(config, self.model)

    def init_asserts(self):
        assert self.config.num_fad_examples % self.config.batch_size == 0, (
            "Number of FAD examples must be divisible by batch size"
        )

    def init_models(self):
        self.encoder = self.config.encoder_class(**self.config.encoder_kwargs).to(
            self.config.device
        )
        self.decoder = self.config.decoder_class(**self.config.decoder_kwargs).to(
            self.config.device
        )
        self.model = FMDiffAE(
            encoder=self.encoder, decoder=self.decoder, **self.config.fmdiffae_kwargs
        ).to(self.config.device)

        self.transform = self.config.transform_class(**self.config.transform_kwargs)

    def init_dataloaders(self):
        super().init_dataloaders()
        if self.config.compute_fad:
            if not hasattr(self.dataset_valid, "valid_vggish_embeddings"):
                raise NotImplementedError(
                    "FAD computation is only supported for datasets with valid_vggish_embeddings"
                )
            self.valid_vggish_embeddings = self.dataset_valid.valid_vggish_embeddings

    def init_save_directories(self):
        super().init_save_directories()
        self.examples_dir = os.path.join(self.save_dir, "examples")
        os.makedirs(self.examples_dir, exist_ok=True)

        if self.config.compute_fad:
            self.fads_path = os.path.join(self.save_dir, "fads.npy")
            self.rec_losses_path = os.path.join(self.save_dir, "rec_losses.npy")

    def init_logging(self):
        super().init_logging()
        if self.config.compute_fad:
            self.fads = []
            self.rec_losses = []

    def load_checkpoint(self):
        super().load_checkpoint()
        if self.config.compute_fad:
            if os.path.exists(self.fads_path):
                self.fads = list(np.load(self.fads_path))
            if os.path.exists(self.rec_losses_path):
                self.rec_losses = list(np.load(self.rec_losses_path))

    def eval_cycle(self):
        super().eval_cycle()
        self.plot_feature_map()
        self.generate_and_save_examples()
        self.generate_and_compute_fad()
        self.save_loss_curves()

        if self.config.compute_fad and len(self.fads) > 0:
            self.save_fad_curves()
            self.save_rec_loss_curves()

    @torch.no_grad()
    def plot_feature_map(self, valid_idx=0):
        curr_step_examples_dir = os.path.join(
            self.examples_dir, f"step={self.step:07d}"
        )
        os.makedirs(curr_step_examples_dir, exist_ok=True)

        model = self.ema_model if self.config.use_ema_weights else self.model
        inputs = self.dataset_valid[valid_idx].to(self.config.device).unsqueeze(0)
        feature_map = model.encoder(inputs).detach().cpu().numpy()[0]
        feature_map_sorted = feature_map[np.argsort(np.mean(feature_map, axis=-1))]

        # Plot Feature Map
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        num_features = feature_map_sorted.shape[0]
        num_frames = feature_map_sorted.shape[1]

        axs[0].imshow(feature_map_sorted, aspect=num_frames / (3 * num_features))
        axs[0].set_title("Feature Map, Sorted by Mean Value")

        # Plot Spectrum
        spectrum = np.fft.rfft(feature_map_sorted, axis=1)
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

        plt.savefig(
            os.path.join(curr_step_examples_dir, "features.png"), bbox_inches="tight"
        )
        plt.close(fig)

    @torch.no_grad()
    def generate_and_save_examples(self, valid_idx=0):
        # Make Directories
        curr_step_examples_dir = os.path.join(
            self.examples_dir, f"step={self.step:07d}"
        )
        curr_step_audio_dir = os.path.join(curr_step_examples_dir, "audio")
        curr_step_npy_dir = os.path.join(curr_step_examples_dir, "npy")
        curr_step_spec_dir = os.path.join(curr_step_examples_dir, "spec")

        os.makedirs(curr_step_audio_dir, exist_ok=True)
        os.makedirs(curr_step_npy_dir, exist_ok=True)
        os.makedirs(curr_step_spec_dir, exist_ok=True)

        # Generate Examples
        num_examples = self.eval_low_highs.shape[0]
        inputs = self.dataset_valid[valid_idx].to(self.config.device)
        inputs = inputs.expand(num_examples, *inputs.shape)
        lows = self.eval_low_highs[:, 0]
        highs = self.eval_low_highs[:, 1]

        model = self.ema_model if self.config.use_ema_weights else self.model

        samples = model.generate(
            inputs=inputs,
            lows=lows,
            highs=highs,
            num_steps=self.config.num_steps_for_example,
        )

        tqdm.write("Inverting Audio")
        audios = self.transform.batched_inverse_transform(samples)

        for i in range(num_examples):
            np.save(
                os.path.join(curr_step_npy_dir, f"{lows[i]:0.3f}_{highs[i]:0.3f}.npy"),
                samples[i].detach().cpu().numpy(),
            )

            save_image(
                os.path.join(curr_step_spec_dir, f"{lows[i]:0.3f}_{highs[i]:0.3f}.png"),
                samples[i],
                title=f"step={self.step}, {lows[i]:0.3f}_{highs[i]:0.3f}.png",
                vmin=-1,
                vmax=1,
            )

        for i in range(num_examples):
            save_audio(
                os.path.join(
                    curr_step_audio_dir, f"{lows[i]:0.3f}_{highs[i]:0.3f}.wav"
                ),
                audios[i],
            )

        del samples, audios
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_and_compute_fad(self):
        model = self.ema_model if self.config.use_ema_weights else self.model

        inputs = self.dataset_valid[: self.config.num_fad_examples]

        current_fads = []
        current_rec_losses = []

        for low, high in self.eval_low_highs:
            samples = torch.cat(
                [
                    model.generate(
                        inputs=inputs[
                            j * self.config.batch_size : (j + 1)
                            * self.config.batch_size
                        ].to(self.config.device),
                        lows=low.expand(self.config.batch_size),
                        highs=high.expand(self.config.batch_size),
                        num_steps=self.config.num_steps_for_fad,
                    )
                    for j in range(
                        self.config.num_fad_examples // self.config.batch_size
                    )
                ],
                dim=0,
            )

            rec_loss = torch.nn.functional.mse_loss(samples.cpu(), inputs)
            current_rec_losses.append(
                [
                    self.step,
                    rec_loss,
                    self.num_epochs_seen,
                    self.total_training_time,
                    self.num_datapoints_seen,
                ]
            )
            tqdm.write(
                f"Reconstruction Loss, low={low:.3f}, high={high:.3f}: {rec_loss:.3f}"
            )

            samples = self.transform.batched_inverse_transform(samples)
            samples = samples - torch.mean(samples, dim=-1, keepdim=True)
            samples = samples / torch.amax(
                torch.abs(samples).clamp(min=1e-8), dim=-1, keepdim=True
            )
            sample_embeddings = get_embeddings_vggish(samples.cpu())
            fad = compute_fad_from_embeddings(
                self.valid_vggish_embeddings, sample_embeddings.numpy()
            )

            current_fads.append(
                [
                    self.step,
                    fad,
                    self.num_epochs_seen,
                    self.total_training_time,
                    self.num_datapoints_seen,
                ]
            )
            tqdm.write(f"FAD, low={low:.3f}, high={high:.3f}: {fad:.3f}")

            del samples
            torch.cuda.empty_cache()

        self.rec_losses.append(current_rec_losses)
        np.save(self.rec_losses_path, self.rec_losses)

        self.fads.append(current_fads)
        np.save(self.fads_path, self.fads)

    def save_fad_curves(self):
        fads = np.array(self.fads)
        for i, (low, high) in enumerate(self.eval_low_highs):
            plt.plot(
                fads[:, i, 0],
                fads[:, i, 1],
                label=f"low={low:.4f}, high={high:.4f}, FAD={fads[-1, i, 1]:.3f}",
            )
        plt.xlabel("Step")
        plt.ylabel("FAD")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "FADs.png"))
        plt.close()

    def save_rec_loss_curves(self):
        rec_losses = np.array(self.rec_losses)
        for i, (low, high) in enumerate(self.eval_low_highs):
            plt.plot(
                rec_losses[:, i, 0],
                rec_losses[:, i, 1],
                label=f"low={low:.4f}, high={high:.4f}, loss={rec_losses[-1, i, 1]:.3f}",
            )
        plt.xlabel("Step")
        plt.ylabel("Rec. Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "Rec_Losses.png"))
        plt.close()
