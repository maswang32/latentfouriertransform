"""User configuration.

Copy this file to `config.py` and fill in your values. `config.py` is
gitignored so your personal paths and API keys never get committed.

Importing this module has the side effect of populating `os.environ` with
the values below. That makes them visible to:
  - wandb (auto-reads WANDB_API_KEY, WANDB_ENTITY, WANDB_PROJECT, WANDB_MODE)
  - Hydra configs (via `${oc.env:VAR_NAME}` interpolation in YAML files)
  - Any Python code that calls `os.environ[...]`

Usage: put `import config` at the top of your entry-point script
(train.py, classification.py, etc.) before anything that reads these values.
"""
import os

# ---- Weights & Biases ----
# Get your key from https://wandb.ai/authorize. Leave WANDB_API_KEY empty
# and run `wandb login` once if you'd rather store it in ~/.netrc.
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_ENTITY"]  = ""
os.environ["WANDB_PROJECT"] = "fmdiffae"
# "online" | "offline" | "disabled"
os.environ["WANDB_MODE"]    = "online"

# ---- Dataset paths ----
# Root directory containing processed datasets. Expected layout:
#
#   <PROCESSED_DATA_DIR>/
#     mtg-jamendo/
#       full-5s/           # training shards + reference VGGish stats
#       full-5s_test/      # held-out evaluation subset
#     gtzan/               # VGGish features + genre labels for classification
#
# The mtg-jamendo/ subfolders are produced by fmdiffae/data/mtg_jamendo.py.
os.environ["PROCESSED_DATA_DIR"] = "/path/to/processed-datasets"

# Raw MTG Jamendo audio (MP3s) -- only needed if you are re-running the
# preprocessing pipeline from scratch.
os.environ["MTG_JAMENDO_RAW_DIR"] = "/path/to/mtg-jamendo/raw"

# The official MTG Jamendo split-0 directory from
# https://github.com/MTG/mtg-jamendo-dataset (clone it and point here):
#   <repo>/data/splits/split-0/
os.environ["MTG_JAMENDO_SPLITS_DIR"] = "/path/to/mtg-jamendo-dataset/data/splits/split-0"

# ---- Experiment outputs ----
# Root directory for training runs. Each run writes to EXP_DIR/runs/<name>/.
os.environ["EXP_DIR"] = "./exp"

# Third-party pretrained checkpoints used by baselines + evaluation in
# reproduce_results/cond_and_blend/generate.py. Expected filenames:
#   rave_pretrained_musicnet.ts   (from the IRCAM RAVE release)
#   BEATs_iter3_plus_AS2M.pt      (from the BEATs repo)
os.environ["PRETRAINED_DIR"] = "./exp/pretrained_checkpoints"
