import os

import numpy as np
import torch
import torchaudio

import dac

import argparse

from beats.BEATs import BEATs, BEATsConfig

from fmdiffae.arc.correlated_fft_mask import CorrelatedFFTMask
from fmdiffae.lightning.lit_fmdiffae import FMDiffAEModule
from fmdiffae.transforms.bigvgan_transform import BigVGANTransform
from fmdiffae.utils.fad import get_embeddings_vggish
from reproduce_results.baselines_and_ablations.unconditional import (
    spectral_guidance,
    dual_spectral_guidance,
)


# Compute All Low_Highs
def get_all_low_highs(mode):
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
        all_low_highs = low_highs_2 + low_highs_4 + low_highs_8
    elif mode == "blend":
        all_low_highs = [
            [low_highs_4[0], low_highs_4[1]],
            [low_highs_4[0], low_highs_4[2]],
            [low_highs_4[0], low_highs_4[3]],
            [low_highs_4[1], low_highs_4[2]],
            [low_highs_4[1], low_highs_4[3]],
            [low_highs_4[2], low_highs_4[3]],
        ]
    else:
        raise ValueError("Mode must be cond or blend")
    return all_low_highs


def get_band_identifier(low_highs, mode):
    if mode == "cond":
        return f"{low_highs[0]:.4f}_{low_highs[1]:.4f}"
    elif mode == "blend":
        return f"{low_highs[0][0]:.4f}_{low_highs[0][1]:.4f} x {low_highs[1][0]:.4f}_{low_highs[1][1]:.4f}"
    else:
        raise ValueError


def main(low_highs, baseline_name, args):
    print(f"Before Expanding Low High {np.array(low_highs).shape}", flush=True)

    # Load Data
    if baseline_name in [
        "fmdiffae_point",
        "fmdiffae_unet",
        "guidance",
        "ilvr",
        "spectrogram",
    ]:
        data_type = "spec"
    elif baseline_name in ["audio", "dac"]:
        data_type = "audio"
    else:
        raise ValueError

    data_path = args.spec_data_path if data_type == "spec" else args.audio_data_path

    data = torch.from_numpy(np.load(data_path))
    inputs = data[: args.num_examples]
    if args.mode == "blend":
        inputs_2 = data[args.num_examples : 2 * args.num_examples]
        inputs = torch.stack((inputs, inputs_2), dim=1)

    # Set up Save Directory
    identifier = get_band_identifier(low_highs, args.mode)
    blend_weights = [0.5, 0.5] if args.mode == "blend" else None

    save_dir = os.path.join(
        args.exp_base_dir, args.exp_name, args.mode, baseline_name, identifier
    )
    os.makedirs(save_dir, exist_ok=True)

    # Make low_highs tensors
    low_highs = [low_highs] * args.num_examples
    lows, highs = torch.tensor(low_highs).unbind(-1)
    print(f"Expanded Low Highs {torch.tensor(low_highs).shape}")
    print(f"{lows.shape=}", flush=True)
    print(f"{highs.shape=}", flush=True)
    print(f"Inputs before selecting baseline {inputs.shape}", flush=True)

    # FMDiffAE Baseline
    if baseline_name in ["fmdiffae_point", "fmdiffae_unet"]:
        ckpt_path = (
            args.fmdiffae_point_ckpt_path
            if baseline_name == "fmdiffae_point"
            else args.fmdiffae_unet_ckpt_path
        )
        # Load Model
        model = FMDiffAEModule.load_torch_model(
            ckpt_path=ckpt_path,
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

    if baseline_name == "audio":
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

    if baseline_name == "spectrogram":
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

    if baseline_name == "dac":
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

    if baseline_name == "guidance":
        # Load Model
        model = FMDiffAEModule.load_torch_model(
            ckpt_path=args.uncond_ckpt_path,
            strict=True,
        ).cuda()

        # Generate
        batched_indices = torch.arange(inputs.shape[0]).split(args.batch_size, dim=0)

        specs = []
        for batch_indices in batched_indices:
            batch_inputs = inputs[batch_indices].cuda()
            batch_lows = lows[batch_indices]
            batch_highs = highs[batch_indices]

            if args.mode == "cond":
                batch_specs = model.generate(
                    batch_size=batch_inputs.shape[0],
                    num_steps=args.num_steps,
                    guidance_fcn=spectral_guidance,
                    guidance_scale=args.guidance_scale,
                    guidance_mode="x0",
                    guidance_lows=lows[batch_indices],
                    guidance_highs=highs[batch_indices],
                    w_iso=0,
                    reference=batch_inputs,
                    w_reference=1,
                    n_fft=batch_inputs.shape[-1],
                ).cpu()
            elif args.mode == "blend":
                batch_specs = model.generate(
                    batch_size=batch_inputs.shape[0],
                    num_steps=args.num_steps,
                    guidance_fcn=dual_spectral_guidance,
                    guidance_scale=args.guidance_scale,
                    guidance_mode="x0",
                    both_guidance_lows=[
                        batch_lows[:, 0],
                        batch_lows[:, 1],
                    ],
                    both_guidance_highs=[
                        batch_highs[:, 0],
                        batch_highs[:, 1],
                    ],
                    references=[batch_inputs[:, 0], batch_inputs[:, 1]],
                    n_fft=batch_inputs.shape[-1],
                ).cpu()
            specs.append(batch_specs)

        specs = torch.cat(specs, dim=0)

        print(f"{specs.shape=}", flush=True)
        torch.save(specs, os.path.join(save_dir, "specs.pt"))

    if baseline_name == "ilvr":
        # Load Model
        model = FMDiffAEModule.load_torch_model(
            ckpt_path=args.uncond_ckpt_path,
            strict=True,
        ).cuda()

        # Generate
        batched_indices = torch.arange(inputs.shape[0]).split(args.batch_size, dim=0)

        specs = []
        for batch_indices in batched_indices:
            batch_inputs = inputs[batch_indices].cuda()
            batch_lows = lows[batch_indices]
            batch_highs = highs[batch_indices]

            if args.mode == "cond":
                batch_specs = model.generate(
                    batch_size=batch_inputs.shape[0],
                    num_steps=args.num_steps,
                    ilvr_mode="cond",
                    ilvr_lows=lows[batch_indices],
                    ilvr_highs=highs[batch_indices],
                    ilvr_reference=batch_inputs,
                    ilvr_nfft=batch_inputs.shape[-1],
                ).cpu()
            elif args.mode == "blend":
                batch_specs = model.generate(
                    batch_size=batch_inputs.shape[0],
                    num_steps=args.num_steps,
                    ilvr_mode="blend",
                    ilvr_lows=[
                        batch_lows[:, 0],
                        batch_lows[:, 1],
                    ],
                    ilvr_highs=[
                        batch_highs[:, 0],
                        batch_highs[:, 1],
                    ],
                    ilvr_reference=[batch_inputs[:, 0], batch_inputs[:, 1]],
                    ilvr_nfft=batch_inputs.shape[-1],
                ).cpu()
            specs.append(batch_specs)

        if baseline_name == "uncondo":
            # Load Model
            model = FMDiffAEModule.load_torch_model(
                ckpt_path=args.uncond_ckpt_path,
                strict=True,
            ).cuda()

            # Generate
            batched_indices = torch.arange(inputs.shape[0]).split(
                args.batch_size, dim=0
            )

            specs = []
            for batch_indices in batched_indices:
                batch_specs = model.generate(
                    batch_size=batch_indices.shape[0],
                    num_steps=args.num_steps,
                ).cpu()
                specs.append(batch_specs)

        specs = torch.cat(specs, dim=0)

        print(f"{specs.shape=}", flush=True)
        torch.save(specs, os.path.join(save_dir, "specs.pt"))

    if data_type == "spec":
        # Invert to Audio
        transform = BigVGANTransform(batch_size=args.transform_batch_size)
        transform.model = transform.model.cuda()
        audios = transform.batched_inverse_transform(
            specs,
            pbar=True,
        )

        print(f"{audios.shape=}", flush=True)

    # Save Audios
    torch.save(audios, os.path.join(save_dir, "audios.pt"))

    if not args.skip_compute_vggish_embeddings:
        # Compute VGGish Embeddings
        vggish_embeddings = get_embeddings_vggish(audios, fs=22050, pbar=True)
        print(f"{vggish_embeddings.shape=}", flush=True)
        torch.save(vggish_embeddings, os.path.join(save_dir, "vggish_embeddings.pt"))

    if args.compute_BEATs_embeddings:
        with torch.no_grad():
            beats_ckpt = torch.load(
                "/data/hai-res/ycda/gen/fmdiffae/reproduce_results/cond_and_blend/exp/diversity/BEATs_iter3_plus_AS2M.pt"
            )
            cfg = BEATsConfig(beats_ckpt["cfg"])
            BEATs_model = BEATs(cfg)
            BEATs_model.load_state_dict(beats_ckpt["model"])
            BEATs_model.eval()
            BEATs_model = BEATs_model.cuda()

            batched_indices = torch.arange(args.num_examples).split(
                args.beats_batch_size, dim=0
            )
            beats_embeddings = torch.zeros(args.num_examples, 296, 768)

            for batch_indices in batched_indices:
                batch_audios = audios[batch_indices].cuda()
                batch_audios = torchaudio.functional.resample(
                    batch_audios, 22050, 16000
                )
                padding_mask = torch.zeros_like(batch_audios).bool()
                representations = BEATs_model.extract_features(
                    batch_audios, padding_mask=padding_mask
                )[0]
                beats_embeddings[batch_indices] = representations.cpu()

            print(f"{beats_embeddings.shape=}", flush=True)
            torch.save(beats_embeddings, os.path.join(save_dir, "beats_embeddings.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name")
    parser.add_argument("baseline_name")
    parser.add_argument("mode")
    parser.add_argument("--low_high_idx", type=int, default=-1)
    parser.add_argument(
        "--exp_base_dir",
        default="/data/hai-res/ycda/gen/fmdiffae/reproduce_results/cond_and_blend/exp/outputs",
    )
    parser.add_argument(
        "--spec_data_path",
        default="/data/hai-res/ycda/processed-datasets/mtg-jamendo/full-5s/valid_subset_spec.npy",
    )
    parser.add_argument(
        "--audio_data_path",
        default="/data/hai-res/ycda/processed-datasets/mtg-jamendo/full-5s/valid_subset_audio.npy",
    )
    parser.add_argument(
        "--fmdiffae_point_ckpt_path",
        default="/data/hai-res/ycda/gen/fmdiffae/exp/runs/point-4gpu-5s-anneal/checkpoints/660000-0.586.ckpt",
    )
    parser.add_argument(
        "--fmdiffae_unet_ckpt_path",
        default="/data/hai-res/ycda/gen/fmdiffae/exp/runs/unet-5s-4gpu-anneal-retry-5/checkpoints/658500-0.802.ckpt",
    )
    parser.add_argument(
        "--uncond_ckpt_path",
        default="/data/hai-res/ycda/gen/fmdiffae/exp/runs/uncondo_anneal_retry/checkpoints/426000-0.500.ckpt",
    )
    parser.add_argument("--num_examples", type=int, default=1024)
    parser.add_argument(
        "--skip_compute_vggish_embeddings", action="store_true", default=False
    )
    parser.add_argument(
        "--compute_BEATs_embeddings", action="store_true", default=False
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--transform_batch_size", type=int, default=128)
    parser.add_argument("--beats_batch_size", type=int, default=128)
    parser.add_argument("--cfg_scale", type=float, default=2.0)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--guidance_scale", type=int, default=1e-3)
    args = parser.parse_args()

    if args.baseline_name == "all":
        list_of_baselines = [
            "audio",
            "dac",
            "guidance",
            "ilvr",
            "fmdiffae_point",
            "fmdiffae_unet",
            "spectrogram",
        ]
    else:
        list_of_baselines = [args.baseline_name]

    for baseline_name in list_of_baselines:
        if args.low_high_idx == -1:
            [
                main(low_highs, baseline_name, args)
                for low_highs in get_all_low_highs(args.mode)
            ]
        else:
            low_highs = get_all_low_highs(args.mode)[args.low_high_idx]
            main(low_highs, baseline_name, args)
