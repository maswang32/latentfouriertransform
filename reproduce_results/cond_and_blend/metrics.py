import numpy as np
import torch

import librosa
import librosa.feature as F
import librosa.onset as O

import os
import json
import argparse

from fmdiffae.arc.correlated_fft_mask import CorrelatedFFTMask
from fmdiffae.utils.fad import compute_fad_from_embeddings
from reproduce_results.cond_and_blend.generate import (
    get_all_low_highs,
    get_band_identifier,
)


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
        x_beat_spec = librosa.autocorrelate(librosa.util.normalize(x_oenv, axis=-1))
        y_beat_spec = librosa.autocorrelate(librosa.util.normalize(y_oenv, axis=-1))
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

        x_in = self.freq_mask(torch.tensor(x_feat), lows=lows, highs=highs).numpy()
        y_in = self.freq_mask(torch.tensor(y_feat), lows=lows, highs=highs).numpy()

        x_out = x_feat - x_in
        y_out = y_feat - y_in

        return metric_fcn(x_in, y_in), metric_fcn(x_out, y_out)

    def compute_blended_error(
        self, x, ref1, ref2, lows1, lows2, highs1, highs2, metric
    ):
        err1 = self.compute_in_and_out_error(x, ref1, lows1, highs1, metric)
        err2 = self.compute_in_and_out_error(x, ref2, lows2, highs2, metric)
        return err1[0], err2[0]


class Aggregator:
    def __init__(
        self,
        exp_dir,
        num_examples=1024,
        ref_audios_path="/data/hai-res/ycda/processed-datasets/mtg-jamendo/full-5s/valid_subset_audio.npy",
        ref_emb_path="/data/hai-res/ycda/processed-datasets/mtg-jamendo/full-5s/valid_vggish_embeddings.npy",
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        fs=22050,
    ):
        self.exp_dir = exp_dir
        self.num_examples = num_examples

        # Prepare Audio Inputs
        all_ref_audios = np.load(ref_audios_path)
        self.ref_audios_cond = all_ref_audios[:num_examples]
        self.ref_audios_blend = np.stack(
            (
                all_ref_audios[:num_examples],
                all_ref_audios[num_examples : 2 * num_examples],
            ),
            axis=1,
        )

        # Get Embeddings
        self.ref_embs = np.load(ref_emb_path)

        self.fe = FeatureExtractor(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length, fs=fs
        )

        # Get Lows/Highs
        self.all_low_highs_cond = get_all_low_highs("cond")
        self.all_low_highs_blend = get_all_low_highs("blend")

    def aggregate_metrics_from_path(
        self,
        mode,
        baseline_name,
        low_highs,
        list_of_metrics,
        save_name=None,
        overwrite=True,
    ):
        results = {}

        # Get Directory containing generations
        identifier = get_band_identifier(low_highs, mode)
        load_dir = os.path.join(self.exp_dir, mode, baseline_name, identifier)

        if save_name is not None:
            json_path = os.path.join(load_dir, f"{save_name}.json")
            if not overwrite and os.path.exists(json_path):
                with open(json_path, "r") as f:
                    results = json.load(f)
                    return results

        # Get Lows/Highs
        lows, highs = torch.tensor(self.num_examples * [low_highs]).unbind(-1)

        # Get Audios
        baseline_audios = torch.load(os.path.join(load_dir, "audios.pt")).numpy()

        for metric in list_of_metrics:
            if mode == "cond":
                errs = self.fe.compute_in_and_out_error(
                    baseline_audios, self.ref_audios_cond, lows, highs, metric
                )
                results[metric] = {"in": errs[0].item(), "out": errs[1].item()}
            elif mode == "blend":
                errs = self.fe.compute_blended_error(
                    x=baseline_audios,
                    ref1=self.ref_audios_blend[:, 0],
                    ref2=self.ref_audios_blend[:, 1],
                    lows1=lows[:, 0],
                    lows2=lows[:, 1],
                    highs1=highs[:, 0],
                    highs2=highs[:, 1],
                    metric=metric,
                )
                results[metric] = {"band1": errs[0].item(), "band2": errs[1].item()}

            print(
                f"{mode} \t {baseline_name} \t {low_highs} \t {metric}: {results[metric]}"
            )

        # Get VGGish Embeddings
        baseline_emb = (
            torch.load(os.path.join(load_dir, "vggish_embeddings.pt"))
            .numpy()
            .reshape(-1, 128)
        )
        results["fad"] = compute_fad_from_embeddings(
            embeddings1=baseline_emb, embeddings2=self.ref_embs
        )
        print(f"{mode} \t {baseline_name} \t {low_highs} \t FAD: {results['fad']}")

        if save_name is not None:
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2)

        return results

    def aggregate_metrics_all(
        self,
        list_of_modes=["blend", "cond"],
        list_of_baselines=[
            "audio",
            "dac",
            "guidance",
            "spectrogram",
            "fmdiffae_point",
            "fmdiffae_unet",
        ],
        list_of_metrics=["loudness", "mcd", "onset", "tonnetz"],
        save_name=None,
        overwrite=True,
    ):
        all_results = {}
        for mode in list_of_modes:
            all_results[mode] = {}

            for baseline_name in list_of_baselines:
                all_results[mode][baseline_name] = {}

                for low_highs in (
                    self.all_low_highs_cond
                    if mode == "cond"
                    else self.all_low_highs_blend
                ):
                    all_results[mode][baseline_name][
                        get_band_identifier(low_highs, mode=mode)
                    ] = self.aggregate_metrics_from_path(
                        mode=mode,
                        baseline_name=baseline_name,
                        low_highs=low_highs,
                        list_of_metrics=list_of_metrics,
                        save_name=save_name,
                        overwrite=overwrite,
                    )

        if save_name is not None:
            out_path = os.path.join(self.exp_dir, f"{save_name}.json")
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

        return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("save_name")
    parser.add_argument("mode")
    parser.add_argument("baseline_name")
    parser.add_argument("--num_examples", type=int, default=1024)
    parser.add_argument(
        "--ref_audios_path",
        default="/data/hai-res/ycda/processed-datasets/mtg-jamendo/full-5s/valid_subset_audio.npy",
    )
    parser.add_argument(
        "--ref_emb_path",
        default="/data/hai-res/ycda/processed-datasets/mtg-jamendo/full-5s/valid_vggish_embeddings.npy",
    )
    parser.add_argument("--n_fft", type=int, default=1024)
    parser.add_argument("--hop_length", type=int, default=256)
    parser.add_argument("--win_length", type=int, default=1024)
    parser.add_argument("--fs", type=int, default=22050)
    args = parser.parse_args()

    ag = Aggregator(
        exp_dir=args.exp_dir,
        num_examples=args.num_examples,
        ref_audios_path=args.ref_audios_path,
        ref_emb_path=args.ref_emb_path,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        win_length=args.win_length,
        fs=args.fs,
    )

    if args.mode == "all":
        list_of_modes = [
            "cond",
            "blend",
        ]
    else:
        list_of_modes = [args.mode]

    if args.baseline_name == "all":
        baseline_names = [
            "guidance",
            "fmdiffae_point",
            "fmdiffae_unet",
            "dac",
            "spectrogram",
            "audio",
        ]
    else:
        baseline_names = [args.baseline_name]

    ag.aggregate_metrics_all(
        list_of_modes=list_of_modes,
        list_of_baselines=baseline_names,
        save_name=args.save_name,
    )
