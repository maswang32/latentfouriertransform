import argparse
import math
import numpy as np
import torch
from fmdiffae.lightning.lit_fmdiffae import FMDiffAEModule


def generate_with_spectral_sweep(
    model,
    window_size,
    step_size,
    batch_size,
    device,
    inputs,
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
    toeplitz = torch.eye(rfft_size).unfold(0, window_size, step_size).sum(dim=-1)

    # Expand to be compatible
    toeplitz = toeplitz.repeat(num_inputs, 1)
    inputs = inputs.repeat_interleave(num_bands, dim=0)

    if init_noise is not None:
        init_noise = init_noise.unsqueeze(0).expand(
            num_inputs * num_bands, *model.datashape
        )

    out = model.batch_generate(
        batch_size=batch_size,
        device=device,
        inputs=inputs,
        fft_mask=toeplitz,
        cfg_scale=cfg_scale,
        init_noise=init_noise,
        num_steps=num_steps,
        outer_pbar=outer_pbar,
        inner_pbar=inner_pbar,
    )
    return out.unflatten(0, (num_inputs, num_bands))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Audio Using Sliding Windows")
    parser.add_argument(
        "ckpt_path",
    )
    parser.add_argument(
        "save_path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=1,
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

    # Load Data
    valid_gtzan_spec = torch.from_numpy(
        np.load("/data/hai-res/ycda/processed-datasets/gtzan/valid_spec.npy")
    )

    model = FMDiffAEModule.load_torch_model(
        ckpt_path=args.ckpt_path,
        strict=True,
    ).cuda()

    model = torch.compile(model)

    gen = torch.Generator()
    gen.manual_seed(3)
    init_noise = torch.randn(*model.datashape, generator=gen) * 80

    out = generate_with_spectral_sweep(
        model,
        window_size=args.window_size,
        step_size=args.step_size,
        batch_size=args.batch_size,
        device=next(model.parameters()).device,
        inputs=valid_gtzan_spec,
        cfg_scale=args.cfg_scale,
        init_noise=init_noise,
        num_steps=args.num_steps,
        outer_pbar=True,
        inner_pbar=False,
    )
    torch.save(out, args.save_path)
