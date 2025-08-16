import argparse
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import essentia.standard as es
from tqdm import tqdm
from fmdiffae.lightning.lit_fmdiffae import FMDiffAEModule
from fmdiffae.transforms.bigvgan_transform import BigVGANTransform
from fmdiffae.utils.fad import get_embeddings_vggish
from fmdiffae.data.data_utils import resample


def nearest_odd(x):
    rounded = round(x)

    if rounded % 2 == 0:
        if x > rounded:
            return rounded + 1
        else:
            return rounded - 1
    else:
        return rounded


def get_linear_sliding_windows(length, window_size, step_size):
    return torch.eye(length).unfold(0, window_size, step_size).sum(dim=-1)


def get_log_sliding_windows(length, width_factor, width_offset, eps, step_size):
    bin_indices = np.arange(length)  # np needed to use round()
    bin_widths = width_factor * np.log(bin_indices + eps) + width_offset

    if not np.all(bin_widths > 0):
        raise ValueError("Bin widths must be > 0")

    windows = []
    for i in range(0, length, step_size):
        quantized_width = nearest_odd(bin_widths[i])
        start = max(i - quantized_width // 2, 0)
        end = min(i + quantized_width // 2 + 1, length)
        m = torch.zeros(length)
        m[start:end] = 1
        windows.append(m)
    return torch.stack(windows)


def windows_to_bins(x, windows):
    windows = windows.to(dtype=x.dtype)
    windows /= windows.sum(dim=0, keepdim=True)

    if x.ndim == 3:
        return torch.einsum("nwd,wb->nbd", x, windows)
    elif x.ndim == 2:
        return torch.einsum("nw,wb->nb", x, windows)
    elif x.ndim == 1:
        return torch.einsum("w,wb->b", x, windows)


def generate_with_spectral_sweep(
    model,
    window_type,
    window_size=None,
    width_factor=None,
    width_offset=None,
    eps=None,
    step_size=None,
    batch_size=None,
    device=None,
    save_path=None,
    save_interval=None,
    inputs=None,
    cfg_scale=3.0,
    init_noise=None,
    num_steps=35,
    outer_pbar=True,
    inner_pbar=True,
):
    rfft_size = model.freq_mask.F

    num_inputs = inputs.shape[0]

    # Create sliding window mask
    if window_type == "linear":
        sliding_window_mask = get_linear_sliding_windows(
            rfft_size, window_size, step_size
        )
    elif window_type == "log":
        sliding_window_mask = get_log_sliding_windows(
            rfft_size,
            width_factor,
            width_offset,
            eps,
            step_size,
        )
    num_bands = sliding_window_mask.shape[0]

    # Expand to be compatible
    sliding_window_mask = sliding_window_mask.repeat(num_inputs, 1)
    inputs = inputs.repeat_interleave(num_bands, dim=0)

    if init_noise is not None:
        init_noise = init_noise.unsqueeze(0).expand(
            num_inputs * num_bands, *model.datashape
        )

    out = model.batch_generate(
        batch_size=batch_size,
        device=device,
        save_path=save_path,
        save_interval=save_interval,
        inputs=inputs,
        fft_mask=sliding_window_mask,
        cfg_scale=cfg_scale,
        init_noise=init_noise,
        num_steps=num_steps,
        outer_pbar=outer_pbar,
        inner_pbar=inner_pbar,
    )
    return out.unflatten(0, (num_inputs, num_bands))


def jensen_shannon_distance(p_logits=None, q_logits=None, p=None, q=None, dim=-1):
    assert (p_logits is None) ^ (p is None)
    assert (q_logits is None) ^ (q is None)

    if p is None:
        p_log = F.log_softmax(p_logits, dim=dim)
    else:
        p_log = p.log()

    if q is None:
        q_log = F.log_softmax(q_logits, dim=dim)
    else:
        q_log = q.log()

    m_log = torch.logaddexp(p_log, q_log) - math.log(2.0)

    # Note - F.kl_div's arguments are (input, target)
    # This is the reverse of the math notation, in which the "true" distribution
    # Is the first argument
    kl1 = F.kl_div(m_log, p_log, log_target=True, reduction="none").sum(dim=dim)
    kl2 = F.kl_div(m_log, q_log, log_target=True, reduction="none").sum(dim=dim)
    return torch.sqrt(0.5 * (kl1 + kl2))


def compute_beat_spectrum(x, fs=22050, hop_length=256, max_size=None):
    oenv = librosa.onset.onset_strength(y=x, sr=fs, hop_length=hop_length)
    autocorr = librosa.autocorrelate(oenv, max_size=max_size)
    return autocorr / np.max(autocorr)


def estimate_tempo(x, fs=22050, hop_length=256):
    oenv = librosa.onset.onset_strength(y=x, sr=fs, hop_length=hop_length)
    tempo = librosa.feature.tempo(onset_envelope=oenv, sr=fs, hop_length=hop_length)
    return tempo.reshape(-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Audio Using Sliding Windows")
    parser.add_argument(
        "save_dir",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--stop_idx",
        type=int,
        default=500,
    )

    # Arguments Related to Spectrogram Generation
    parser.add_argument(
        "--skip_spec_generation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--input_spec_path",
        default=None,
    )
    parser.add_argument(
        "--ckpt_path",
        default=None,
    )
    parser.add_argument(
        "--same_init_noise",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=35,
    )

    # Arguments Related to Windowing
    parser.add_argument(
        "--window_type",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--width_factor",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--width_offset",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
    )

    # Arguments Related to Spectrogram to Audio Inversion
    parser.add_argument(
        "--skip_inversion",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--transform_batch_size",
        type=int,
        default=128,
    )
    # Arguments Related to Onset Envelope
    parser.add_argument(
        "--skip_onset_envelope",
        action="store_true",
        default=False,
    )
    # Arguments Related to Tempo Estimation
    parser.add_argument(
        "--skip_estimate_tempo",
        action="store_true",
        default=False,
    )
    # Arguments Related to VGGish Embeddings
    parser.add_argument(
        "--skip_compute_vggish_embeddings",
        action="store_true",
        default=False,
    )
    # Arguments Related to Tonnetz
    parser.add_argument(
        "--skip_tonnetz",
        action="store_true",
        default=False,
    )
    # Arguments Related to Pitch Detection
    parser.add_argument(
        "--skip_pitch_detect",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if not args.skip_spec_generation:
        # Load Data/Model
        input_spec = torch.from_numpy(np.load(args.input_spec_path))

        model = FMDiffAEModule.load_torch_model(
            ckpt_path=args.ckpt_path,
            strict=True,
        ).cuda()

        model = torch.compile(model)

        if args.same_init_noise:
            gen = torch.Generator()
            gen.manual_seed(3)
            init_noise = torch.randn(*model.datashape, generator=gen) * 80
        else:
            init_noise = None

        for i in range(args.start_idx, args.stop_idx):
            specs = generate_with_spectral_sweep(
                model,
                window_type=args.window_type,
                window_size=args.window_size,
                width_factor=args.width_factor,
                width_offset=args.width_offset,
                eps=args.eps,
                step_size=args.step_size,
                batch_size=args.batch_size,
                device=next(model.parameters()).device,
                save_path=None,
                save_interval=None,
                inputs=input_spec[i].unsqueeze(0),
                cfg_scale=args.cfg_scale,
                init_noise=init_noise,
                num_steps=args.num_steps,
                outer_pbar=True,
                inner_pbar=False,
            )
            i_path = os.path.join(args.save_dir, f"{i:04d}")
            os.makedirs(i_path, exist_ok=True)
            torch.save(specs.squeeze(0), os.path.join(i_path, "specs.pt"))

    if not args.skip_inversion:
        transform = BigVGANTransform(batch_size=args.transform_batch_size)
        transform.model = transform.model.cuda()

        for i in range(args.start_idx, args.stop_idx):
            i_path = os.path.join(args.save_dir, f"{i:04d}")
            specs = torch.load(os.path.join(i_path, "specs.pt"))
            print(f"{specs.shape=}")
            audios = transform.batched_inverse_transform(
                specs,
                pbar=True,
            )
            print(f"{audios.shape=}")
            torch.save(audios, os.path.join(i_path, "audios.pt"))

    if not args.skip_onset_envelope:
        for i in tqdm(
            range(args.start_idx, args.stop_idx),
            desc="Computing Onset Envelopes",
        ):
            i_path = os.path.join(args.save_dir, f"{i:04d}")
            audios = torch.load(os.path.join(i_path, "audios.pt")).numpy()
            oenvs = librosa.onset.onset_strength(y=audios, sr=22050, hop_length=256)
            np.save(os.path.join(i_path, "oenvs.npy"), oenvs)

    if not args.skip_estimate_tempo:
        for i in tqdm(
            range(args.start_idx, args.stop_idx),
            desc="Estimating Tempos",
        ):
            i_path = os.path.join(args.save_dir, f"{i:04d}")
            audios = torch.load(os.path.join(i_path, "audios.pt")).numpy()
            tempos = estimate_tempo(audios)
            np.save(os.path.join(i_path, "tempos.npy"), tempos)

    if not args.skip_compute_vggish_embeddings:
        for i in range(args.start_idx, args.stop_idx):
            i_path = os.path.join(args.save_dir, f"{i:04d}")
            audios = torch.load(os.path.join(i_path, "audios.pt"))
            vggish_embeddings = get_embeddings_vggish(audios, fs=22050, pbar=True)
            torch.save(vggish_embeddings, os.path.join(i_path, "vggish_embeddings.pt"))

    if not args.skip_tonnetz:
        for i in tqdm(
            range(args.start_idx, args.stop_idx),
            desc="Computing Tonnetz",
        ):
            i_path = os.path.join(args.save_dir, f"{i:04d}")
            audios = torch.load(os.path.join(i_path, "audios.pt")).numpy()
            tonnetz = librosa.feature.tonnetz(
                y=audios, sr=22050, hop_length=256
            )  # Default Hop Length
            np.save(os.path.join(i_path, "tonnetz.npy"), tonnetz)

    if not args.skip_pitch_detect:
        loudness_eq = es.EqualLoudness(sampleRate=44100)
        pitch_estimator = es.PredominantPitchMelodia(sampleRate=44100)

        for i in tqdm(
            range(args.start_idx, args.stop_idx),
            desc="Estimating Pitches",
        ):
            i_path = os.path.join(args.save_dir, f"{i:04d}")
            audios = torch.load(os.path.join(i_path, "audios.pt"))
            audios_resampled = resample(audios.cuda(), 22050, 44100).cpu().numpy()

            audio_normalized = np.stack([loudness_eq(x) for x in audios_resampled])

            pitches = []
            confidences = []

            for x in audio_normalized:
                p, c = pitch_estimator(x)
                pitches.append(p)
                confidences.append(c)

            pitches = np.stack(pitches)
            confidences = np.stack(confidences)
            np.save(os.path.join(i_path, "pitches.npy"), pitches)
            np.save(os.path.join(i_path, "pitch_confidences.npy"), confidences)
