import os
import numpy as np
import torch
import torchaudio
import glob
import webdataset as wds
from webdataset import ShardWriter
from tqdm import tqdm
from fmdiffae.transforms.bigvgan_transform import BigVGANTransform
from torch.utils.data.dataset import Dataset


def resample(x, fs_orig, fs_target):
    return torchaudio.functional.resample(
        x,
        orig_freq=fs_orig,
        new_freq=fs_target,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )


def chunk_audio(
    audio_path,
    chunk_length_samples,
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
        Tensor (N, chunk_length_samples): audio chunks
    """
    x, fs_orig = torchaudio.load(audio_path)

    # Resample if needed
    if fs_target is not None and fs_orig != fs_target:
        x = resample(x, fs_orig=fs_orig, fs_target=fs_target)

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
    chunks = chunks[chunk_energies >= energy_threshold]

    # Return Normalized Chunks
    return chunks / chunks.abs().amax(-1, keepdim=True).clamp_min(1e-8)


def save_webdataset(
    audio_paths,
    audio_names,
    save_dir,
    maxcount,  # for 256 x 80 specs + 65536 audio, 8192 is 2.8 GB/shard
    shuffle=True,
    random_seed=7,
    pattern="data-%06d.tar",
    transform_cls=BigVGANTransform,
    transform_kwargs=None,
    chunk_audio_kwargs=None,
):
    assert len(audio_paths) == len(audio_names), "num. paths and names must match"

    # Default Arguments
    transform_kwargs = transform_kwargs or {}
    chunk_audio_kwargs = chunk_audio_kwargs or {}

    indices = np.arange(len(audio_paths))

    if shuffle:
        np.random.default_rng(random_seed).shuffle(indices)

    print(indices)
    with ShardWriter(os.path.join(save_dir, pattern), maxcount=maxcount) as sink:
        transform = transform_cls(**transform_kwargs)

        for i in tqdm(indices.tolist(), desc="Writing Chunks"):
            chunks = chunk_audio(audio_paths[i], **chunk_audio_kwargs)

            # Filter Out Empty Chunks
            print(f"\n{i}\t{chunks.shape}\t{audio_names[i]}", flush=True)
            if chunks.numel() == 0:
                continue

            specs = transform(chunks)
            chunks = chunks.numpy()
            specs = specs.numpy()

            for j, (chunk, spec) in enumerate(zip(chunks, specs)):
                key = f"{audio_names[i]}_{j:05d}"
                sink.write({"__key__": key, "audio.npy": chunk, "spec.npy": spec})


def get_webdataset(
    split,
    base_dir,
    data_type,
    shuffle_size,
):
    shard_paths = sorted(glob.glob(os.path.join(base_dir, split, "data-*.tar")))

    if split == "train":
        dataset = wds.WebDataset(
            shard_paths, resampled=True, nodesplitter=wds.split_by_node
        ).shuffle(shuffle_size)
    else:
        dataset = wds.WebDataset(
            shard_paths, resampled=False, nodesplitter=wds.split_by_node
        )

    dataset = dataset.decode()

    if data_type == "both":
        dataset = dataset.to_tuple("audio.npy", "spec.npy").map_tuple(
            torch.from_numpy, torch.from_numpy
        )
    else:
        dataset = (
            dataset.to_tuple(f"{data_type}.npy")
            .map_tuple(torch.from_numpy)
            .map(lambda x: x[0])
        )

    return dataset


class SingleTensorDataset(Dataset):
    def __init__(self, path):
        super().__init__()
        self.data = torch.from_numpy(np.load(path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
