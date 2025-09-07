import os

import numpy as np
import torch
import torchaudio

import dac

import argparse

from fmdiffae.arc.correlated_fft_mask import CorrelatedFFTMask
from fmdiffae.lightning.lit_fmdiffae import FMDiffAEModule
from fmdiffae.transforms.bigvgan_transform import BigVGANTransform
from fmdiffae.utils.fad import get_embeddings_vggish


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


def main(low_highs, args):
    print(f"Before Expanding Low High {np.array(low_highs).shape}", flush=True)

    # Load Data
    data = torch.from_numpy(np.load(args.data_path))
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
    else:
        raise ValueError

    save_dir = os.path.join(args.exp_dir, args.mode, args.baseline_name, identifier)
    os.makedirs(save_dir, exist_ok=True)

    # Make low_highs tensors
    low_highs = [low_highs] * args.num_examples
    lows, highs = torch.tensor(low_highs).unbind(-1)
    print(f"Expanded Low Highs {torch.tensor(low_highs).shape}")
    print(f"{lows.shape=}", flush=True)
    print(f"{highs.shape=}", flush=True)
    print(f"Inputs before selecting baseline {inputs.shape}", flush=True)

    # FMDiffAE Baseline
    if args.baseline_name in ["fmdiffae_unet", "fmdiffae_point"]:
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

        print(f"{specs.shape=}", flush=True)

        # Invert, Save Audio
        transform = BigVGANTransform(batch_size=args.transform_batch_size)
        transform.model = transform.model.cuda()
        audios = transform.batched_inverse_transform(
            specs,
            pbar=True,
        )

        print(f"{audios.shape=}", flush=True)

    if args.baseline_name == "audio":
        print(f"Audio Before Resampling: {inputs.shape}", flush=True)
        inputs = torchaudio.functional.resample(inputs, 256, 1).unsqueeze(-2)
        print(f"Audio After Resampling {inputs.shape}", flush=True)
        freq_mask = CorrelatedFFTMask(n_fft=inputs.shape[-1])

        if args.mode == "cond":
            out = freq_mask(inputs, lows=lows, highs=highs)
        elif args.mode == "blend":
            out = (
                freq_mask(inputs[:, 0], lows=lows[:, 0], highs=highs[:, 0])
                * blend_weights[0]
                + freq_mask(inputs[:, 1], lows=lows[:, 1], highs=highs[:, 1])
                * blend_weights[1]
            )
        else:
            raise ValueError

        print(f"{out.shape=}", flush=True)

        audios = torchaudio.functional.resample(out, 1, 256).squeeze(-2)

        print(f"{audios.shape=}", flush=True)

    if args.baseline_name == "spectrogram":
        freq_mask = CorrelatedFFTMask(n_fft=inputs.shape[-1])

        if args.mode == "cond":
            specs = freq_mask(inputs, lows=lows, highs=highs)
        elif args.mode == "blend":
            specs = (
                freq_mask(inputs[:, 0], lows=lows[:, 0], highs=highs[:, 0])
                * blend_weights[0]
                + freq_mask(inputs[:, 1], lows=lows[:, 1], highs=highs[:, 1])
                * blend_weights[1]
            )
        else:
            raise ValueError

        print(f"Spectrogram: {specs.shape}", flush=True)
        torch.save(specs, os.path.join(save_dir, "specs.pt"))

        transform = BigVGANTransform(batch_size=args.transform_batch_size)
        transform.model = transform.model.cuda()
        audios = transform.batched_inverse_transform(
            specs,
            pbar=True,
        )

        print(f"{audios.shape=}", flush=True)

    if args.baseline_name == "dac":
        with torch.no_grad():
            dac_model_path = dac.utils.download(model_type="44khz")
            dac_model = dac.DAC.load(dac_model_path).cuda()
            inputs = torchaudio.functional.resample(inputs, 22050, 44100)
            inputs = inputs.unsqueeze(-2)  # Need Channel Dim
            inputs = dac_model.preprocess(inputs, 44100)

            if args.mode == "blend":
                inputs = inputs.flatten(0, 1)

            print(f"DAC {inputs.shape=}")

            all_zs = torch.cat(
                [dac_model.encode(x[None].cuda())[0].cpu() for x in inputs], dim=0
            )

            print(f"{all_zs.shape=}")

            freq_mask = CorrelatedFFTMask(n_fft=all_zs.shape[-1])

            if args.mode == "cond":
                all_zs = freq_mask(all_zs, lows=lows, highs=highs)

            elif args.mode == "blend":
                print(f"blend, before unflatten {all_zs.shape=}")
                all_zs = all_zs.unflatten(0, (-1, 2))

                print(f"blend, after unflatten {all_zs.shape=}")
                all_zs = (
                    freq_mask(all_zs[:, 0], lows=lows[:, 0], highs=highs[:, 0])
                    * blend_weights[0]
                    + freq_mask(all_zs[:, 1], lows=lows[:, 1], highs=highs[:, 1])
                    * blend_weights[1]
                )
            print(f"after fftmask {all_zs.shape=}")
            all_outs = torch.cat(
                [dac_model.decode(x[None].cuda()).cpu() for x in all_zs], dim=0
            )
            print(f"after fftmask {all_outs.shape=}")

            audios = torchaudio.functional.resample(all_outs, 44100, 22050)
            print(f"before squeeze {audios.shape=}")

            audios = audios.squeeze(-2)
            print(f"after squeeze {audios.shape=}")

    # Save Audios
    torch.save(audios, os.path.join(save_dir, "audios.pt"))

    # Compute VGGish Embeddings
    vggish_embeddings = get_embeddings_vggish(audios, fs=22050, pbar=True)
    print(f"{vggish_embeddings.shape=}", flush=True)
    torch.save(vggish_embeddings, os.path.join(save_dir, "vggish_embeddings.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_dir")
    parser.add_argument("baseline_name")
    parser.add_argument("mode")
    parser.add_argument("low_high_idx", type=int)
    parser.add_argument("ckpt_path")
    parser.add_argument("data_path")
    parser.add_argument("--num_examples", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--transform_batch_size", type=int, default=128)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--num_steps", type=int, default=100)
    args = parser.parse_args()

    if args.low_high_idx == -1:
        [main(low_highs, args) for low_highs in get_low_highs(args.mode)]
    else:
        low_highs = get_low_highs(args.mode)[args.low_high_idx]
        main(low_highs, args)
