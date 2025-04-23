import math
import torch
from bigvgan import bigvgan
from bigvgan.meldataset import mel_spectrogram
from tqdm import tqdm

"""
Big VGAN model transform.
"""


class BigVGANTransform:
    def __init__(
        self,
        model_name="bigvgan_v2_22khz_80band_256x",
        load_model_on_init=True,
        batch_size=256,
        idle_device="cpu",
        active_device="cuda",
        max_log_spec_value=2.1922,
        min_log_spec_value=-11.5129,  # torch.log(1e-5)
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.idle_device = idle_device
        self.active_device = active_device
        self.max_log_spec_value = max_log_spec_value
        self.min_log_spec_value = min_log_spec_value
        self.range = max_log_spec_value - min_log_spec_value

        if model_name == "bigvgan_v2_22khz_80band_256x":
            self.n_fft = 1024
            self.num_mels = 80
            self.sampling_rate = 22050
            self.hop_size = 256
            self.win_size = 1024
            self.fmin = 0
        else:
            raise NotImplementedError

        if load_model_on_init:
            self.load_model()

    def load_model(self):
        self.model = bigvgan.BigVGAN.from_pretrained(
            "nvidia/" + self.model_name, use_cuda_kernel=False
        )
        self.model.remove_weight_norm()
        self.model.to(self.idle_device).eval()

    def __call__(self, x):
        """
        Notes:
            - input to mel_spectrogram must be 2D, [B, T]
            - mel_spectrogram gives [B, num_mels, num_frames]
        """
        log_spec = mel_spectrogram(
            x.view(-1, x.shape[-1]),
            n_fft=self.n_fft,
            num_mels=self.num_mels,
            sampling_rate=self.sampling_rate,
            hop_size=self.hop_size,
            win_size=self.win_size,
            fmin=self.fmin,
            device=x.device,
            dtype=x.dtype,
        )
        log_spec = log_spec.reshape(*x.shape[:-1], self.num_mels, -1)
        return 2 * ((log_spec - self.max_log_spec_value) / (self.range)) + 1

    def batched_inverse_transform(self, x):
        self.model = self.model.to(self.active_device).eval()
        x = x.to(self.active_device)

        num_batches = math.ceil(x.shape[0] / self.batch_size)
        if num_batches > 1:
            inverted = []
            for chunk in tqdm(
                torch.split(x, self.batch_size), desc="Inverting", leave=False
            ):
                inverted.append(self.inverse_transform(chunk))
            inverted = torch.cat(inverted, dim=0)

        else:
            inverted = self.inverse_transform(x)
        self.model = self.model.to(self.idle_device)

        return inverted

    def inverse_transform(self, x):
        x = ((x - 1) / 2) * self.range + self.max_log_spec_value
        with torch.inference_mode():
            x_3d = x.view(-1, x.shape[-2], x.shape[-1])
            inverted = self.model(x_3d)
            return inverted.reshape(*x.shape[:-2], -1)
