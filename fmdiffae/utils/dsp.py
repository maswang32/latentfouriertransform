import torchaudio


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
