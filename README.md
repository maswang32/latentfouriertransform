# Latent Fourier Transform

Code for [**Latent Fourier Transform**][project-page] (ICLR 2026, **oral**)
by [Mason L. Wang](https://masonlwang.com/) and
[Cheng-Zhi Anna Huang](https://czhuang.github.io/) (CSAIL, MIT).

- Project page + audio demos: <https://masonlwang.com/latentfouriertransform/>
- Paper (OpenReview): <https://openreview.net/forum?id=ogMxCjdCCq>
<!-- TODO: fill in once the arXiv link is live -->
- arXiv: <https://arxiv.org/abs/TODO>

[project-page]: https://masonlwang.com/latentfouriertransform/

## What is this?

LATENTFT is a framework that provides **frequency-domain controls in latent
space** for generative music models. It combines a diffusion autoencoder
with a Fourier transform applied to the latent time series, so that
different musical patterns end up at different latent frequencies — i.e.
separated by *timescale*. Masking those latent frequencies during training
yields representations that can be manipulated coherently at inference,
allowing us to:

- Generate variations of a song while preserving patterns at chosen
  timescales.
- Blend two songs, picking which timescales come from each.
- "Zoom in" on selected latent frequencies and hear the musical patterns
  they encode in isolation.

Conceptually, LATENTFT acts as an **equalizer for musical structure**,
complementing the traditional audible-frequency equalizer.

## Repo contents

This repo contains two things:

- **`latentft/`** — the core library. The Frequency-Masked Diffusion
  AutoEncoder (FMDiffAE) model, EDM-style sampler, UNet / pointwise encoders,
  correlated FFT mask module, BigVGAN mel-spectrogram transform, VGGish-based
  FAD utilities, and Lightning modules for training.
- **`reproduce_results/`** — the scripts, Hydra configs, and entry points
  needed to regenerate every table and figure in the paper. See the
  [Reproducing paper results](#reproducing-paper-results) section below.

## Install

```bash
git clone https://github.com/maswang32/latentfouriertransform.git
cd latentfouriertransform

# Recommended: a fresh environment
conda create -n latentft python=3.11 -y
conda activate latentft

pip install -e .
```

This gives you everything needed to **train the model and use the
`latentft/` library** (FMDiffAE, the EDM sampler, BigVGAN transforms,
VGGish FAD, etc.).

If you also want to **reproduce the paper's experiments**, install the
extras:

```bash
pip install -e ".[reproduce_results]"
```

The extras (`librosa`, `essentia`, `descript-audio-codec`, `mir_eval`)
are imported only from scripts under `reproduce_results/` — specifically
for evaluation metrics, baseline models (DAC, RAVE, VampNet), and the
sweep / demo scripts. You don't need them for `train.py`.

BigVGAN is installed from a lightly-modified fork
(`git+https://github.com/maswang32/BigVGAN.git`) that's already listed in
`setup.py`. No manual step needed.

## Configuration

Personal paths and Weights & Biases credentials live in a gitignored
`config.py`. Copy the template and edit it:

```bash
cp config.example.py config.py
# then open config.py and fill in the values
```

`config.py` sets environment variables (`WANDB_API_KEY`, `PROCESSED_DATA_DIR`,
`EXP_DIR`, etc.) that both W&B and the Hydra configs (via `${oc.env:VAR}`)
pick up automatically. Entry-point scripts (`train.py`, `classification.py`,
`generate.py`, …) import `config` at the top, so setting values there is all
you need to do.

If you'd rather not store the W&B key in a file, leave `WANDB_API_KEY`
blank in `config.py` and run `wandb login` once.

## Data

### MTG Jamendo (main experiments)

1. Clone <https://github.com/MTG/mtg-jamendo-dataset> and download the audio
   following their instructions. Point `MTG_JAMENDO_RAW_DIR` at the audio
   root and `MTG_JAMENDO_SPLITS_DIR` at `<repo>/data/splits/split-0`.
2. Run the preprocessing script, which resamples to 22050 Hz, chunks into
   131072-sample (~5.9 s) clips, writes WebDataset shards, and precomputes
   VGGish statistics for FAD:

   ```bash
   python latentft/data/mtg_jamendo.py full-5s
   ```

   Output goes under `$PROCESSED_DATA_DIR/mtg-jamendo/full-5s/`. Re-running
   with a different `save_name` and flags (e.g. `--only_inst_tagged`,
   `--exclude_voice`) produces sibling subsets.

### GTZAN (classification only)

The classification experiments train a linear probe on top of frozen
features. You'll need a `$PROCESSED_DATA_DIR/gtzan/` directory with
pre-extracted features and genre labels. A preprocessing script for this is
not yet included in this release — see the [Caveats](#caveats) section.

## Training

Training is driven by Hydra. The main config is
[`exp/configs/default.yaml`](exp/configs/default.yaml):
350k steps of bf16-mixed DDP training on MTG Jamendo 5s with the 5s
FMDiffAE model, Adam at 1e-4, and W&B logging.

```bash
# Single-GPU debug run
python train.py name=my_first_run

# Multi-GPU (launched by Lightning DDP)
python train.py name=my_big_run trainer.devices=4
```

Hydra lets you override anything from the command line, e.g.:

```bash
python train.py \
    name=unet-5s-4gpu \
    trainer.devices=4 \
    trainer.max_steps=660000 \
    batch_size=512
```

Checkpoints and W&B artifacts are written to
`$EXP_DIR/runs/<name>/`. If training is interrupted, re-running the same
command auto-resumes from `checkpoints/last.ckpt`.

## Reproducing paper results

> **Note:** The scripts in this section require the `[reproduce_results]`
> extras (see [Install](#install)). If you only want to train or use the
> core model, skip this section.

Every script below assumes `config.py` is in place and the MTG Jamendo
preprocessing has been run. Checkpoints produced by `train.py` (or the
corresponding baseline trainers) are passed in explicitly via CLI flags —
we intentionally don't hardcode checkpoint paths.

### Conditional generation and blending (main results + ablations)

`reproduce_results/cond_and_blend/generate.py` runs the full generation
pipeline across the paper's baselines and ablations. It takes an experiment
name, a baseline name (or a group like `all` / `ablations` / `rebuttals`),
and a conditioning mode:

```bash
python reproduce_results/cond_and_blend/generate.py \
    <exp_name> <baseline_name> <mode> \
    --latentft_point_ckpt_path  $EXP_DIR/runs/.../660000-0.586.ckpt \
    --latentft_unet_ckpt_path   $EXP_DIR/runs/.../658500-0.802.ckpt \
    --uncond_ckpt_path          $EXP_DIR/runs/.../...ckpt
    # ...etc; see --help for the full list of --*_ckpt_path flags
```

Supported `baseline_name` values include `latentft_point`, `latentft_unet`,
`latentft_bandpass`, `unconditional`, `dac`, `vampnet`, `ilvr`, `guidance`,
`cross`, `spectrogram`, the ablations (`abl_freq_masking`, `abl_corr`,
`abl_log_scale`, `abl_spec_encoder`, `abl_no_encoder`), plus the group
aliases `all`, `ablations`, and `rebuttals`.

Metrics are computed by
[`reproduce_results/cond_and_blend/metrics.py`](reproduce_results/cond_and_blend/metrics.py)
(FAD, loudness correlation, beat-spectrum cosine, tonnetz distance, and
the in/out-of-band variants used in the paper).

### Classification (linear probe)

A linear probe on GTZAN over frozen features:

```bash
python reproduce_results/classification/classification.py name=probe_run
```

Default config lives in
[`reproduce_results/classification/exp/configs/default.yaml`](reproduce_results/classification/exp/configs/default.yaml)
(12k steps, Adam, batch 8192). Override W&B project, data, or model the
same way as for the main trainer, e.g.
`python reproduce_results/classification/classification.py logger.project=my_project`.

### Unconditional & no-encoder baselines

These live as standalone training scripts under
`reproduce_results/baselines_and_ablations/`:

- `unconditional.py` — plain EDM diffusion trained on mels (no latent).
- `no_encoder.py` — FMDiffAE with the encoder replaced by a deterministic
  downsampled-audio feature, for the "no learned encoder" ablation.
- `cross_synthesis.py` — cepstral cross-synthesis helper used for the
  `cross` baseline.

### Hyperparameter sweeps

`reproduce_results/sweep/sweep.py` drives the sweep over latent dimension,
masking thresholds, etc. See its `--help` for knobs.

### Demos & figures

`reproduce_results/demos/eq_plots.py` reproduces the EQ / blending figures.

## Repository layout

```
.
├── latentft/                    # Core library
│   ├── arc/                     # UNet, pointwise net, correlated FFT mask
│   ├── data/                    # WebDataset utils + MTG Jamendo preprocessor
│   ├── diffusion/fmdiffae.py    # The FMDiffAE module + EDM sampler
│   ├── lightning/               # Lightning modules, callbacks, data module
│   ├── transforms/              # BigVGAN mel transform
│   └── utils/fad.py             # VGGish FAD
├── reproduce_results/
│   ├── baselines_and_ablations/ # Standalone training scripts
│   ├── classification/          # GTZAN linear probe
│   ├── cond_and_blend/          # generate.py + metrics.py (main eval)
│   ├── demos/                   # Figure-generation scripts
│   └── sweep/                   # Hyperparameter sweeps
├── exp/configs/                 # Hydra configs for train.py
├── train.py                     # Main training entry point
├── config.example.py            # Template for user config (copy to config.py)
└── setup.py
```

## Caveats

- **GTZAN preprocessing script is not included yet.** The classification
  config expects a pre-built `$PROCESSED_DATA_DIR/gtzan/` directory with
  VGGish features and labels. This will be added in a follow-up.
- **A small number of baseline Hydra configs are not in this repo.** The
  `dac_frontend`, `abl_dft`, and a couple of other "rebuttal" ablations
  reference configs that are not currently checked in. These baselines are
  otherwise fully implemented; the missing configs will be restored in a
  follow-up.
- **Checkpoint paths in `generate.py` are intentionally required CLI
  arguments.** The pre-cleanup code hardcoded specific cluster paths; we
  removed them so other users wouldn't silently pick up broken defaults.

## Notes for adapting to new datasets

- Recalibrate the chunking **energy threshold** in
  `latentft/data/data_utils.py::chunk_audio`. The default (0.003) is
  tuned for MTG Jamendo at 22050 Hz; other corpora / sample rates may
  need a different value.
- Recalibrate the **BigVGAN mel-spectrogram max** used for log-scale
  normalization in `latentft/transforms/bigvgan_transform.py`.

## Citation

```bibtex
@inproceedings{wang2026latentft,
  title     = {Latent {F}ourier Transform},
  author    = {Wang, Mason L. and Huang, Cheng-Zhi Anna},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {Oral presentation},
  url       = {https://openreview.net/forum?id=ogMxCjdCCq}
}
```

## License

Released under the [MIT License](LICENSE).
