import numpy as np

import librosa
import librosa.feature as F
import librosa.onset as O

from fmdiffae.utils.fad import compute_fad_from_embeddings


class FeatureExtractor:
    """
    Everything Outputs N, C, T
    """

    def __init__(self, n_fft=1024, hop_length=256, win_length=1024, fs=22050):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.fs = fs

        self._pad_amount = (win_length - hop_length) // 2

    def loudness(self, x):
        x = self._pad(x)
        pow_spec = (
            np.abs(
                librosa.stft(
                    y=x,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    win_length=self.win_length,
                    center=False,
                )
            )
            ** 2
        )

        weights_db = librosa.A_weighting(
            librosa.fft_frequencies(n_fft=self.n_fft, sr=self.fs)
        )
        weights = 10 ** (weights_db / 10)
        power_per_frame = np.mean(pow_spec * weights[..., None], axis=-2, keepdims=True)
        integrated_loudness = 10 * np.log10(power_per_frame)
        return integrated_loudness

    def mfcc(self, x):
        x = self._pad(x)
        return F.mfcc(
            y=x,
            sr=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False,
        )

    def onset_strength(self, x):
        # First Frames are always 0
        return O.onset_strength(
            y=x,
            sr=self.fs,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )[..., None, 1:]

    def tonnetz(self, x):
        x = self._pad(x, use_centered=True)
        return F.tonnetz(
            y=x,
            sr=self.fs,
            hop_length=self.hop_length,
        )[..., 1:]

    def _pad(self, x, use_centered=False):
        if use_centered:
            p_left = self._pad_amount + self.hop_length - self.win_length // 2
            p_right = 0
        else:
            p_left = self._pad_amount
            p_right = self._pad_amount
        return np.pad(
            x,
            pad_width=((0, 0), (p_left, p_right)),
            mode="constant",
            constant_values=0.0,
        )
