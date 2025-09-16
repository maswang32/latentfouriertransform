import torch


def get_spectral_envelope(
    x,
    cutoff_bin=50,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
):
    spec = torch.log(
        torch.abs(
            torch.stft(
                x,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                onesided=False,
                return_complex=True,
            )
        )
        + 1e-7
    )
    # IFFT of Real, Even should be real
    ceps = torch.fft.ifft(spec, dim=-2).real
    ceps[..., cutoff_bin + 1 : n_fft - cutoff_bin, :] = 0
    ceps[..., cutoff_bin, :] *= 0.5
    ceps[..., n_fft - cutoff_bin, :] *= 0.5

    # Symmetric Filtering should result in a real inversion
    return torch.exp(torch.fft.fft(ceps, dim=-2).real)


def get_cross_synthesis(
    carrier,
    modulator,
    cutoff_bin=100,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
):
    carrier_spec = torch.stft(
        carrier,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        onesided=False,
        return_complex=True,
    )
    carrier_env = get_spectral_envelope(
        carrier,
        cutoff_bin=cutoff_bin,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )
    print(f"{carrier_spec.shape=}")
    print(f"{carrier_env.shape=}")

    flattened_carrier_spec = carrier_spec / (carrier_env + 1e-7)

    modulator_env = get_spectral_envelope(
        modulator,
        cutoff_bin=cutoff_bin,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
    )

    modulated_carrier_spec = modulator_env * flattened_carrier_spec

    return torch.istft(
        modulated_carrier_spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        onesided=False,
        return_complex=False,
    )
