import argparse
import os
import math
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from fmdiffae.lightning.lit_fmdiffae import FMDiffAEModule


def get_sliding_window_mask(length, window_size, step_size):
    return torch.eye(length).unfold(0, window_size, step_size).sum(dim=-1)


def generate_with_spectral_sweep(
    model,
    window_size,
    step_size,
    batch_size,
    device,
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
    num_bands = math.ceil((rfft_size - window_size + 1) / step_size)

    # Create sliding window mask
    sliding_window_mask = get_sliding_window_mask(rfft_size, window_size, step_size)

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


def jensen_shannon_distance(p_logits, q_logits, dim=-1):
    p_log = F.log_softmax(p_logits, dim=dim)
    q_log = F.log_softmax(q_logits, dim=dim)

    m_log = torch.logaddexp(p_log, q_log) - math.log(2.0)

    # Note - F.kl_div's arguments are (input, target)
    # This is the reverse of the math notation, in which the "true" distribution
    # Is the first argument
    kl1 = F.kl_div(m_log, p_log, log_target=True, reduction="none").sum(dim)
    kl2 = F.kl_div(m_log, q_log, log_target=True, reduction="none").sum(dim)
    return torch.sqrt(0.5 * (kl1 + kl2))


def compute_beat_spectrum(x, fs=22050, hop_length=256, max_size=None):
    oenv = librosa.onset.onset_strength(y=x, sr=fs, hop_length=hop_length)
    autocorr = librosa.autocorrelate(oenv, max_size=max_size)
    return autocorr / np.max(autocorr)


def estimate_tempo(x, fs=22050, hop_length=256):
    oenv = librosa.onset.onset_strength(y=x, sr=fs, hop_length=hop_length)
    tempo = librosa.feature.tempo(onset_envelope=oenv, sr=fs, hop_length=hop_length)
    return tempo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Audio Using Sliding Windows")
    parser.add_argument(
        "ckpt_path",
    )
    parser.add_argument(
        "save_dir",
    )
    parser.add_argument(
        "--same_init_noise",
        action="store_true",
        default=False
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
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
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
    args = parser.parse_args()

    # Load Data/Model
    valid_gtzan_spec = torch.from_numpy(
        np.load("/data/hai-res/ycda/processed-datasets/gtzan/valid_spec.npy")[args.start_idx:args.stop_idx]
    )
    
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

    for i, spec in valid_gtzan_spec:
        out = generate_with_spectral_sweep(
            model,
            window_size=args.window_size,
            step_size=args.step_size,
            batch_size=args.batch_size,
            device=next(model.parameters()).device,
            save_path=None,
            save_interval=None,
            inputs=spec.unsqueeze(0),
            cfg_scale=args.cfg_scale,
            init_noise=init_noise,
            num_steps=args.num_steps,
            outer_pbar=True,
            inner_pbar=False,
        )
        torch.save(out, os.path.join(args.save_dir, "{i}.pt"))
