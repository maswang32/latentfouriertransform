import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from IPython.display import Audio, display


def save_audio(
    path, x, fs=22050, num_channels=1, demean=True, normalize=True, show_audio=False
):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    x = x.detach().cpu().reshape(num_channels, -1)

    if demean:
        x = x - torch.mean(x)
    if normalize:
        x = x / torch.max(torch.abs(x))

    if show_audio:
        display(Audio(x, rate=fs))

    torchaudio.save(path, x, fs)


def save_image(path, x, title, vmin=None, vmax=None, colorbar=False):
    x = x.detach().cpu().numpy()
    x = x.reshape(-1, x.shape[-1])
    plt.imshow(x, vmin=vmin, vmax=vmax)

    if colorbar:
        plt.colorbar()

    plt.title(title)
    plt.savefig(path)
    plt.close()
