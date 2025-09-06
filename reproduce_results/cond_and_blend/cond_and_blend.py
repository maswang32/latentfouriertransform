import os

import numpy as np
import torch

import argparse

from fmdiffae.lightning.lit_fmdiffae import FMDiffAEModule
from fmdiffae.transforms.bigvgan_transform import BigVGANTransform


# Compute Low_Highs
def get_low_highs(mode):
    vs = [
        0.0000,
        0.0078,
        0.0157,
        0.0313,
        0.0626,
        0.1251,
        0.2501,
        0.5001,
        1.0000,
    ]
    low_highs_2 = [
        [vs[0], vs[4]],
        [vs[4], vs[8]],
    ]
    low_highs_4 = [
        [vs[0], vs[2]],
        [vs[2], vs[4]],
        [vs[4], vs[6]],
        [vs[6], vs[8]],
    ]
    low_highs_8 = [
        [vs[0], vs[1]],
        [vs[1], vs[2]],
        [vs[2], vs[3]],
        [vs[3], vs[4]],
        [vs[4], vs[5]],
        [vs[5], vs[6]],
        [vs[6], vs[7]],
        [vs[7], vs[8]],
    ]

    if mode == "cond":
        low_highs = low_highs_2 + low_highs_4 + low_highs_8
    elif mode == "blend":
        low_highs = [
            [low_highs_4[0], low_highs_4[1]],
            [low_highs_4[0], low_highs_4[2]],
            [low_highs_4[0], low_highs_4[3]],
            [low_highs_4[1], low_highs_4[2]],
            [low_highs_4[1], low_highs_4[3]],
            [low_highs_4[2], low_highs_4[3]],
        ]
    else:
        raise ValueError("Mode must be cond or blend")
    return low_highs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("baseline_name")
    parser.add_argument("mode")
    parser.add_argument("low_high_idx")
    parser.add_argument("ckpt_path")
    parser.add_argument("data_path")
    parser.add_argument("--num_examples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--transform_batch_size", type=int, default=128)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--num_steps", type=int, default=100)
    args = parser.parse_args()

    # Get low_highs
    low_highs = get_low_highs(args.mode)[args.low_high_idx]
    print(f"{np.array(low_highs.shape)=}")

    # Load Data
    data = torch.from_numpy(np.from_numpy(args.data_path))
    inputs = data[: args.num_examples]
    if args.mode == "blend":
        inputs_2 = data[args.num_examples : 2 * args.num_examples]
        inputs = torch.stack((inputs, inputs_2), dim=1)

    # Set up Save Directory
    if args.mode == "cond":
        identifier = f"{low_highs[0]:.4f}_{low_highs[1]:.4f}"
        blend_weights = None
    elif args.mode == "blend":
        identifier = f"{low_highs[0][0]:.4f}_{low_highs[0][1]:.4f} x {low_highs[1][0]:.4f}_{low_highs[1][1]:.4f}"
        blend_weights = [0.5, 0.5]

    save_dir = os.path.join(args.exp_dir, args.baseline_name, identifier)
    os.makedirs(save_dir, exist_ok=True)

    # FMDiffAE Baseline
    if args.baseline_name == "ours":
        # Make low_highs tensors
        low_highs = [low_highs] * args.num_examples
        lows, highs = torch.tensor(low_highs).unbind(-1)

        print(f"{torch.tensor(low_highs).shape}")
        print(f"{lows.shape}")
        print(f"{highs.shape}")

        # Load Model
        model = FMDiffAEModule.load_torch_model(
            ckpt_path=args.ckpt_path,
            strict=True,
        ).cuda()

        # Generate
        specs = model.batch_generate(
            batch_size=args.batch_size,
            device=next(model.parameters()).device,
            save_path=os.path.join(save_dir, "specs.pt"),
            inputs=inputs,
            lows=lows,
            highs=highs,
            cfg_scale=args.cfg_scale,
            blend_weights=blend_weights,
            num_steps=args.num_steps,
        )

        print(f"{specs.shape=}")

        # Invert, Save Audio
        transform = BigVGANTransform(batch_size=args.transform_batch_size)
        transform.model = transform.model.cuda()
        audios = transform.batched_inverse_transform(
            specs,
            pbar=True,
        )

        print(f"{audios.shape=}")

        torch.save(audios, os.path.join(save_dir, "audios.pt"))
