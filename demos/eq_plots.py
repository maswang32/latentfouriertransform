import numpy as np
import torch

import librosa
import scipy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import subprocess

from typing import List, Optional
from fmdiffae.lightning.lit_fmdiffae import FMDiffAEModule
from fmdiffae.transforms.bigvgan_transform import BigVGANTransform


LATENT_DEFAULTS = {
    "x_label": "Latent Frequency (Hz)",
    "x_eps": 0.5,
    "x_lim": [0, 22050 / 512],
    "x_num_pts": 513,
    "x_ticks": [0, 1, 2, 5, 10, 20, 40],
    "x_tick_labels": [0, 1, 2, 5, 10, 20, 40],
    "y_label": "Weighting",
    "y_lim": (0, 1.1),
    "y_ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "y_tick_labels": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "band_transition_width": 0.1,
}


class Track:
    def __init__(
        self,
        name: str,
        low_highs: List[List[float]],
        floor_ceilings: Optional[List[List[float]]] = None,
        color: str = "#DF00FE",
        duration: float = 131072 / 22050,
        transition_duration: float = 1.0,
        alpha: float = 1.0,
        linewidth: float = 3,
        start_time: float = 0.0,
        amp_curves: Optional[np.ndarray] = None,  # N, F
    ):
        self.name = name
        self.low_highs = low_highs
        if floor_ceilings is None:
            self.floor_ceilings = [[0, 1]] * len(low_highs)
        else:
            assert len(low_highs) == len(floor_ceilings)
            self.floor_ceilings = floor_ceilings

        self.color = color
        self.duration = duration
        self.transition_duration = transition_duration

        self.alpha = alpha
        self.linewidth = linewidth
        self.start_time = start_time

        self.amp_curves = amp_curves


class BandpassAnimation:
    FONT_PARAMS = {
        "font.family": "cmb10",
        "font.weight": "bold",
    }
    DPI = 300
    FPS = 30
    BITRATE = 1800

    FACE_COLOR = "black"
    SPINE_COLOR = "#333"
    LABEL_FONT_SIZE = 24
    LABEL_COLOR = "#FFF"
    TICK_FONT_SIZE = 18
    TICK_COLOR = "#888"

    GRID_COLOR = "#FFF"
    GRID_ALPHA = 0.1

    LEGEND_LOC = "upper right"
    LEGEND_FRAME_ALPHA = 0.9
    LEGEND_FACE_COLOR = "black"
    LEGEND_EDGE_COLOR = "#444"
    LEGEND_FONT_SIZE = 20
    LEGEND_LABEL_COLOR = "#FFF"

    def __init__(
        self,
        tracks: List[Track],
        x_label,
        x_eps,
        x_lim,
        x_num_pts,
        x_ticks,
        x_tick_labels,
        y_label,
        y_lim,
        y_ticks,
        y_tick_labels,
        band_transition_width=0.3,
        figsize=(16, 9),
    ):
        self.tracks = tracks

        self.x_label = x_label
        self.x_eps = x_eps
        self.x_display = np.linspace(
            self.to_display(x_lim[0]),
            self.to_display(x_lim[1]),
            x_num_pts,
        )
        self.x_values = self.to_values(self.x_display)
        self.x_ticks = x_ticks
        self.x_tick_labels = x_tick_labels

        self.y_label = y_label
        self.y_lim = y_lim
        self.y_ticks = y_ticks
        self.y_tick_labels = y_tick_labels

        self.band_transition_width = band_transition_width
        self.figsize = figsize

    def setup_plot(self):
        with matplotlib.rc_context(self.FONT_PARAMS):
            self.all_elements = {}
            self.fig = plt.figure(figsize=self.figsize, facecolor=self.FACE_COLOR)
            self.ax = self.fig.add_subplot(111, facecolor=self.FACE_COLOR)
            self.ax.set_xlabel(
                self.x_label, color=self.LABEL_COLOR, fontsize=self.LABEL_FONT_SIZE
            )
            self.ax.set_ylabel(
                self.y_label, color=self.LABEL_COLOR, fontsize=self.LABEL_FONT_SIZE
            )
            self.ax.spines["bottom"].set_color(self.SPINE_COLOR)
            self.ax.spines["left"].set_color(self.SPINE_COLOR)
            self.ax.spines["top"].set_visible(False)
            self.ax.spines["right"].set_visible(False)
            self.ax.tick_params(colors=self.TICK_COLOR)
            self.ax.set_xlim(self.x_display[0], self.x_display[-1])
            self.ax.set_ylim(self.y_lim)

            display_ticks = self.to_display(np.array(self.x_ticks))
            self.ax.set_xticks(display_ticks)
            self.ax.set_xticklabels(self.x_tick_labels, fontsize=self.TICK_FONT_SIZE)
            self.ax.set_yticks(self.y_ticks)
            self.ax.set_yticklabels(self.y_tick_labels, fontsize=self.TICK_FONT_SIZE)
            self.ax.grid(
                True, which="both", alpha=self.GRID_ALPHA, color=self.GRID_COLOR
            )

            for i, track in enumerate(self.tracks):
                track_elements = {}

                (line,) = self.ax.plot(
                    [],
                    [],
                    color=track.color,
                    linewidth=track.linewidth,
                    alpha=track.alpha,
                    label=track.name,
                    zorder=3 + i * 2,
                )

                track_elements["line"] = line
                glow_lines = []
                for j in range(2):
                    alpha = track.alpha * 0.15 * (1 - j / 2)
                    width = track.linewidth + (j + 1) * 3
                    (glow_line,) = self.ax.plot(
                        [],
                        [],
                        color=track.color,
                        linewidth=width,
                        alpha=alpha,
                        zorder=2 + i * 2,
                    )
                    glow_lines.append(glow_line)

                track_elements["glow_lines"] = glow_lines

                # Fill
                track_elements["fill"] = None

                self.all_elements[track.name] = track_elements

            if len(self.tracks) >= 1:
                self.ax.legend(
                    loc=self.LEGEND_LOC,
                    framealpha=self.LEGEND_FRAME_ALPHA,
                    facecolor=self.LEGEND_FACE_COLOR,
                    edgecolor=self.LEGEND_EDGE_COLOR,
                    fontsize=self.LEGEND_FONT_SIZE,
                    labelcolor=self.LEGEND_LABEL_COLOR,
                )
        plt.tight_layout()

    def to_display(self, x):
        return np.log(x + self.x_eps)

    def to_values(self, x):
        return np.exp(x) - self.x_eps

    def smooth_step(self, edge0, edge1, x):
        # Assigns values from 0-1 to x values within edges
        t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        # Cubic Interpolation
        return t * t * (3.0 - 2.0 * t)

    def calculate_curve(self, low, high, floor, ceiling):
        if high - low < 1e-6:
            return np.full_like(self.x_values, floor)

        low_d = self.to_display(low)
        high_d = self.to_display(high)

        low_trans_start = low_d - self.band_transition_width
        high_trans_end = high_d + self.band_transition_width
        step_up = self.smooth_step(low_trans_start, low_d, self.x_display)
        step_down = self.smooth_step(high_d, high_trans_end, self.x_display)
        weights = step_up * (1.0 - step_down)
        return floor + (ceiling - floor) * weights

    def get_track_state_at_time(self, track, time):
        track_time = time - track.start_time
        if track_time < 0:
            return [0, 0], [0, 0]

        band_idx = int(track_time // track.duration)

        num_tracks = len(track.low_highs)

        if band_idx >= num_tracks:
            band_idx = num_tracks - 1

        time_in_cycle = track_time - track.duration * band_idx

        # Interpolate Amplitude Curves
        if track.amp_curves is not None:
            n_amp_frames = track.amp_curves.shape[-1]
            # Temporal Endpoints align to first and last frames
            t_amp = (time_in_cycle / track.duration) * (n_amp_frames - 1)
            segment_idx = int(t_amp)
            t_between = t_amp - segment_idx
            prev_amp = track.amp_curves[band_idx, segment_idx]

            # Guards aganest case where t_amp = n_amp_frames
            if t_between > 0:
                next_amp = track.amp_curves[band_idx, segment_idx + 1]
                amp = prev_amp + (next_amp - prev_amp) * t_between
            else:
                amp = prev_amp
        else:
            amp = None

        # Manage Transitions between low/highs, and ceilings
        if time_in_cycle < track.transition_duration and band_idx > 0:
            transition_progress = time_in_cycle / track.transition_duration
            t = self.smooth_step(0, 1, transition_progress)

            # Interpolate Bands
            prev_band = np.array(track.low_highs[band_idx - 1])
            curr_band = np.array(track.low_highs[band_idx])
            interpolated_band = prev_band + (curr_band - prev_band) * t

            # Interpolate Floor-Ceilings
            prev_floor_ceiling = np.array(track.floor_ceilings[band_idx - 1])
            curr_floor_ceiling = np.array(track.floor_ceilings[band_idx])
            interpolated_floor_ceiling = (
                prev_floor_ceiling + (curr_floor_ceiling - prev_floor_ceiling) * t
            )
            return interpolated_band, interpolated_floor_ceiling, amp
        else:
            return track.low_highs[band_idx], track.floor_ceilings[band_idx], amp

    def update(self, frame):
        curr_time = frame / self.FPS
        artists = []

        for track in self.tracks:
            track_elements = self.all_elements[track.name]
            (low, high), (floor, ceiling), amp = self.get_track_state_at_time(
                track, curr_time
            )
            curve = self.calculate_curve(low, high, floor, ceiling)

            # Update
            track_elements["line"].set_data(self.x_display, curve)
            artists.append(track_elements["line"])

            for glow_line in track_elements["glow_lines"]:
                glow_line.set_data(self.x_display, curve)
                artists.append(glow_line)

            if amp is not None:
                prev_fill = track_elements.get("fill")
                if prev_fill is not None:
                    prev_fill.remove()

                new_fill = self.ax.fill_between(
                    self.x_display,
                    self.y_lim[0],
                    # curve,
                    np.minimum(amp, curve),
                    color=track.color,
                    alpha=np.clip(amp, 0, 1),
                    zorder=1,
                )
                track_elements["fill"] = new_fill
                artists.append(new_fill)

        return artists

    def animate(self, duration, save_path=None):
        self.setup_plot()
        total_frames = int(duration * self.FPS)

        anim = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=total_frames,
            blit=True,
            repeat=False,
        )

        if save_path:
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=self.FPS, bitrate=self.BITRATE)
            anim.save(save_path, writer=writer)
            print(f"Animation saved to {save_path}")
        else:
            plt.show()


class AudioGenerator:
    def __init__(
        self,
        ckpt_path,
        device,
    ):
        self.model = FMDiffAEModule.load_torch_model(ckpt_path=ckpt_path).to(device)
        self.device = device

        self.transform = BigVGANTransform()
        self.transform.model = self.transform.model.to(device)

    @torch.no_grad()
    def generate_bandpass(
        self,
        spec,
        low_highs,
        cfg_scale=1.0,
        floor=0.1,
        ceiling=0.9,
        init_noise=None,
        num_steps=100,
        normalize_output=False,
        guidance_scale=None,
        w_iso=0,
        w_reference=1e-4,
        use_inversion=False,
    ):
        num = len(low_highs)
        lows, highs = torch.tensor(low_highs).unbind(dim=-1)

        lows = torch.stack((torch.zeros(num), lows), dim=-1)
        highs = torch.stack((torch.ones(num), highs), dim=-1)

        inputs = spec.expand(num, 2, -1, -1).to(self.device)
        blend_weights = [floor, ceiling]

        guidance_args = {}
        if guidance_scale is not None:
            z_reference = self.model.encoder(spec.unsqueeze(0).to(self.device)).detach()
            guidance_args.update(
                guidance_fcn=self.latent_spectral_guidance,
                guidance_scale=guidance_scale,
                guidance_mode="x0",
                guidance_lows=lows.view(-1),
                guidance_highs=highs.view(-1),
                w_iso=w_iso,
                z_reference=z_reference,
                w_reference=w_reference,
            )

        if use_inversion:
            assert init_noise is None, "Cannot provide init_noise if doing inversion"
            init_noise = self.model.generate(
                inputs=spec.unsqueeze(0).to(self.device),
                lows=[0.7],  # No lows or highs
                highs=[0.3],
                num_steps=35,
                invert=True,
            )

        out = self.model.generate(
            inputs=inputs,
            lows=lows,
            highs=highs,
            cfg_scale=cfg_scale,
            blend_weights=blend_weights,
            init_noise=init_noise,
            num_steps=num_steps,
            pbar=True,
            **guidance_args,
        )

        if normalize_output:
            out = out / torch.amax(torch.abs(out), dim=(-2, -1), keepdim=True)

        audios = self.transform.batched_inverse_transform(out, pbar=True).cpu()
        return audios

    @torch.no_grad()
    def generate_blend(
        self,
        specs,
        low_highs,
        cfg_scale=1.0,
        blend_weights=[0.5, 0.5],
        init_noise=None,
        num_steps=100,
        normalize_output=False,
    ):
        """
        specs: (num_to_blend, ...)
        low_highs: (N, num_to_blend, 2)
        blend_weights: (N, num_to_blend) OR (num_to_blend,)
        """
        num = len(low_highs)

        blend_weights = torch.tensor(blend_weights)
        if blend_weights.ndim == 1:
            blend_weights = blend_weights.expand(num, -1)
        num_to_blend = blend_weights.shape[-1]

        low_highs = torch.tensor(low_highs)

        # (N, num_to_blend)
        lows, highs = low_highs.unbind(dim=-1)

        # (N, num_to_blend, *datashape)
        inputs = specs.expand(num, num_to_blend, -1, -1).to(self.device)

        out = self.model.generate(
            inputs=inputs,
            lows=lows,
            highs=highs,
            cfg_scale=cfg_scale,
            blend_weights=blend_weights,
            init_noise=init_noise,
            num_steps=num_steps,
            pbar=True,
        )

        if normalize_output:
            out = out / torch.amax(torch.abs(out), dim=(-2, -1), keepdim=True)

        audios = self.transform.batched_inverse_transform(out, pbar=True).cpu()
        return audios

    @torch.no_grad()
    def generate_conditional(
        self,
        specs,
        low_highs,
        cfg_scale=1.0,
        init_noise=None,
        num_steps=100,
        normalize_output=False,
    ):
        """
        inputs: (*datashape) OR (N, *datashape)
        specs: (N, ...)
        low_highs: (N, 2)
        """
        low_highs = torch.tensor(low_highs)
        lows, highs = low_highs.unbind(dim=-1)

        inputs = specs.expand(low_highs.shape[0], -1, -1).to(self.device)

        out = self.model.generate(
            inputs=inputs,
            lows=lows,
            highs=highs,
            cfg_scale=cfg_scale,
            init_noise=init_noise,
            num_steps=num_steps,
            pbar=True,
        )

        if normalize_output:
            out = out / torch.amax(torch.abs(out), dim=(-2, -1), keepdim=True)

        audios = self.transform.batched_inverse_transform(out, pbar=True).cpu()
        return audios

    def latent_spectral_guidance(
        self,
        x,
        guidance_lows,
        guidance_highs,
        w_iso=0,
        z_reference=None,
        w_reference=1e-4,
    ):
        n_fft = self.model.freq_mask.n_fft
        F = n_fft // 2 + 1
        v = torch.linspace(0, 1, F)

        # Encode x
        z = self.model.encoder(x)

        # Select spectrum inside selected band
        fft_mask = (v >= guidance_lows.unsqueeze(1)) & (
            v <= guidance_highs.unsqueeze(1)
        )
        fft_mask = fft_mask.unsqueeze(-2).to(device=z.device, dtype=z.dtype)
        inv_fft_mask = 1 - fft_mask

        z_spectrum = torch.fft.rfft(z, n=n_fft)

        # Isolation Loss
        z_power_spectrum = torch.abs(z_spectrum) ** 2
        loss_iso = torch.sum(inv_fft_mask * z_power_spectrum)

        # Reference Loss
        if z_reference is not None:
            ref_spectrum = torch.fft.rfft(z_reference, n=n_fft)
            squared_errors = torch.abs(ref_spectrum - z_spectrum) ** 2
            loss_reference = torch.sum(fft_mask * squared_errors)
        else:
            loss_reference = 0

        return w_iso * loss_iso + w_reference * loss_reference


def mux(video_path, audio_path, out_path):
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-preset",
        "veryfast",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "160k",
        "-shortest",
        "-movflags",
        "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def compute_loudness_curve(
    x, n_fft=2048, hop_length=256, fs=22050, win_std_s=0.05, win_length=51
):
    pow_spec = (
        np.abs(librosa.stft(y=x, n_fft=n_fft, hop_length=hop_length, center=True)) ** 2
    )

    weights = librosa.A_weighting(librosa.fft_frequencies(n_fft=n_fft, sr=fs))
    weights = 10 ** (weights / 10)
    power_per_frame = np.mean(pow_spec * weights[..., None], axis=-2)
    integrated_loudness = 10 * np.log10(power_per_frame)

    win_std_frames = (win_std_s * fs) / hop_length

    kernel = scipy.signal.windows.gaussian(win_length, std=win_std_frames)
    kernel = kernel.reshape((1,) * (integrated_loudness.ndim - 1) + (-1,))
    return scipy.signal.convolve(integrated_loudness, kernel, mode="same")


def adjust_loudness_curve(x, pow=0.5, eps=1e-7):
    x = x - np.min(x) + eps
    return (x / np.max(x)) ** pow
