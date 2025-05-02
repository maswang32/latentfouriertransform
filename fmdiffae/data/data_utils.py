import os
import glob
import torch
import torchaudio
import webdataset as wds
from fmdiffae.transforms.bigvgan import BigVGANTransform


def resample(x, orig_rate, target_rate):
    return torchaudio.functional.resample(
        x,
        orig_rate,
        target_rate,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )


def chunk_audio(
    audio_path,
    chunk_length_samples=65536,
    fs_target=22050,
    energy_threshold=0.003,
):
    """
    Args:
        audio_path (str or PathLike): Path to audio file.
        chunk_length_samples (int): length of chunks in samples
        fs_target (float): rate to resample to, if needed
        energy_threshold (float):
            For chunks of length 65536, fs=22050, 0.003 is a safe threshold.
            Below 0.002 we get incomplete clips.

    Returns:
        Tensor: (N, chunk_length_samples). audio chunks
    """
    x, fs = torchaudio.load(audio_path)

    # Resample if needed
    if fs_target is not None and fs != fs_target:
        x = resample(x, fs, fs_target)

    # Demean and normalize
    x = x - torch.mean(x, dim=-1, keepdim=True)
    x = x / x.abs().amax(-1, keepdim=True).clamp_min(1e-8)

    # Split into chunks
    num_chunks = x.shape[-1] // chunk_length_samples
    x = x[..., : num_chunks * chunk_length_samples]
    chunks = x.reshape(-1, chunk_length_samples)  # Flatten Channels

    # Demean chunks
    chunks = chunks - torch.mean(chunks, dim=-1, keepdim=True)

    # Threshold based on energy
    chunk_energies = torch.mean(chunks**2, dim=-1)
    return chunks[chunk_energies >= energy_threshold]


def save_webdataset(
    audio_paths,
    audio_names,
    save_dir,
    maxcount=12000,  # Approx 1 GB / File for 256 x 80 specs
    audio_pattern="audio-%06d.tar",
    specs_pattern="specs-%06d.tar",
    transform_cls=BigVGANTransform,
    transform_kwargs=None,
    chunk_audio_kwargs=None,
):
    # Default Arguments
    transform_kwargs = transform_kwargs or {"load_model_on_init": False}
    chunk_audio_kwargs = chunk_audio_kwargs or {}

    audio_sink = wds.ShardWriter(
        os.path.join(save_dir, audio_pattern), maxcount=maxcount
    )
    spec_sink = wds.ShardWriter(
        os.path.join(save_dir, specs_pattern), maxcount=maxcount
    )
    transform = transform_cls(**transform_kwargs)

    for audio_path, audio_name in zip(audio_paths, audio_names):
        chunks = chunk_audio(audio_path, **chunk_audio_kwargs)
        specs = transform(chunks)

        for i, (chunk, spec) in enumerate(zip(chunks, specs)):
            key = f"{audio_name}_{i:05d}"

            audio_sink.write({"__key__": key, "audio.npy": chunk})
            spec_sink.write({"__key__": key, "spec.pt": spec})

    audio_sink.close()
    spec_sink.close()
