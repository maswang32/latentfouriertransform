import os
import numpy as np
import torch
from torch.utils.data import Dataset


class VoiceSpectrogramDataset(Dataset):
    def __init__(
        self,
        split="train",
        data_dir="/scratch/ycda/datasets/Voice/normalized_22khz_65536",
    ):
        super().__init__()
        self.split = split
        self.data_dir = data_dir

        # Load Dataset
        self.indices = np.load(os.path.join(data_dir, f"{split}_indices.npy"))
        self.specs = np.load(os.path.join(data_dir, "specs.npy"), mmap_mode="r")

        if split == "valid":
            self.valid_vggish_embeddings = np.load(
                os.path.join(data_dir, "valid_vggish_embeddings.npy"), mmap_mode="r"
            )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return torch.from_numpy(self.specs[self.indices[idx]].copy())
