import numpy as np
import torch

import librosa
import librosa.feature as F
import librosa.onset as O

from fmdiffae.arc.correlated_fft_mask import CorrelatedFFTMask
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
        self.freq_mask = CorrelatedFFTMask(n_fft=self.n_fft)

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

    def cosine_similarity(self, x, y):
        """
        N x T
        """
        numerator = np.sum(x * y, axis=-1)
        norm_1 = np.linalg.norm(x, axis=-1)
        norm_2 = np.linalg.norm(y, axis=-1)
        denom = np.clip(norm_1 * norm_2, 1e-7, None)
        return numerator / denom

    def loudness_correlation(self, x_loudness, y_loudness):
        """
        N x 1 x T
        """
        x_loudness = x_loudness.squeeze(axis=1)  # N, T
        y_loudness = y_loudness.squeeze(axis=1)  # N, T
        x_demeaned = x_loudness - np.mean(x_loudness, axis=-1, keepdims=True)  # N, T
        y_demeaned = y_loudness - np.mean(y_loudness, axis=-1, keepdims=True)  # N, T
        return np.mean(self.cosine_similarity(x_demeaned, y_demeaned))

    def mcd(self, x_mfcc, y_mfcc):
        alpha = (10 * np.sqrt(2)) / np.log(10)

        # Sum over channels
        sum_sq = np.sum((x_mfcc - y_mfcc) ** 2, axis=-2)
        return alpha * np.mean(np.sqrt(sum_sq))

    def beat_spectral_similarity(self, x_oenv, y_oenv):
        x_oenv = x_oenv.squeeze(axis=1)
        y_oenv = y_oenv.squeeze(axis=1)
        x_beat_spec = librosa.autocorrelate(librosa.utils.normalize(x_oenv))
        y_beat_spec = librosa.autocorrelate(librosa.utils.normalize(y_oenv))
        return np.mean(self.cosine_similarity(x_beat_spec, y_beat_spec))

    def tonnetz_distance(self, x_tonnetz, y_tonnetz):
        """
        N x 6 x T
        """
        return np.linalg.norm(x_tonnetz - y_tonnetz, axis=-2).mean()

    def compute_in_and_out_error(self, x, y, lows, highs, metric):
        if metric == "loudness":
            x_feat = self.loudness(x)
            y_feat = self.loudness(y)
            metric_fcn = self.loudness_correlation
        elif metric == "mcd":
            x_feat = self.mfcc(x)
            y_feat = self.mfcc(y)
            metric_fcn = self.mcd
        elif metric == "onset":
            x_feat = self.onset_strength(x)
            y_feat = self.onset_strength(y)
            metric_fcn = self.beat_spectral_similarity
        elif metric == "tonnetz":
            x_feat = self.tonnetz(x)
            y_feat = self.tonnetz(y)
            metric_fcn = self.tonnetz_distance
        else:
            raise NotImplementedError(f"{metric} not implemented")

        x_in = self.freq_mask(x_feat, lows=lows, highs=highs)
        y_in = self.freq_mask(y_feat, lows=lows, highs=highs)

        x_out = x_feat - x_in
        y_out = y_feat - y_in

        return metric_fcn(x_in, y_in), metric_fcn(x_out, y_out)

    def compute_blended_error(
        self, x, ref1, ref2, lows1, lows2, highs1, highs2, metric
    ):
        err1 = self.compute_in_and_out_error(x, ref1, lows1, highs1, metric)
        err2 = self.compute_in_and_out_error(x, ref2, lows2, highs2, metric)
        return err1[0], err2[0]


def compute_fad_from_paths(
    target_emb_path,
    ref_emb_path="/data/hai-res/ycda/processed-datasets/mtg-jamendo/full-5s/valid_vggish_embeddings.npy",
):
    targ_emb = torch.load(target_emb_path).numpy().reshape(-1, 128)
    ref_emb = np.load(ref_emb_path).reshape(-1, 128)
    return compute_fad_from_embeddings(embeddings1=targ_emb, embeddings2=ref_emb)
