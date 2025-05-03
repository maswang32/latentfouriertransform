import torch
from torchvggish import vggish, vggish_input
from scipy import linalg
import numpy as np
from tqdm import tqdm


def get_embeddings_vggish(x, fs=22050, pbar=False):
    model = vggish()
    model.eval()
    model.postprocess = False
    embeddings = []

    with torch.no_grad():
        if pbar:
            iterator = tqdm(x, desc="Computing VGGish Embeddings", leave=False)
        else:
            iterator = x

        for example in iterator:
            embeddings.append(
                model.forward(
                    vggish_input.waveform_to_examples(
                        example.numpy().reshape(-1), sample_rate=fs
                    )
                )
            )
    return torch.stack(embeddings, dim=0)


def compute_fad_from_embeddings(
    embeddings1=None, embeddings2=None, mean1=None, mean2=None, sigma1=None, sigma2=None
):
    if mean1 is None:
        mean1 = np.mean(embeddings1, axis=0)
    if sigma1 is None:
        sigma1 = np.cov(embeddings1, rowvar=False)
    if mean2 is None:
        mean2 = np.mean(embeddings2, axis=0)
    if sigma2 is None:
        sigma2 = np.cov(embeddings2, rowvar=False)

    covmean = linalg.sqrtm(sigma1.dot(sigma2).astype(complex))
    if not np.isfinite(covmean).all():
        print("Adding 1e-6 to diagonal of covariance estimates")
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset).astype(complex))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            im = np.max(np.abs(covmean.imag))
            print(f"Warning: Imaginary Component in Covmean {im}")
        covmean = covmean.real

    return (
        np.sum((mean1 - mean2) ** 2)
        + np.trace(sigma1)
        + np.trace(sigma2)
        - 2 * np.trace(covmean)
    )
