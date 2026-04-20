"""
Copy this file to `config.py` and fill in your values.

`config.py` is gitignored so personal info doesn't get committed

Hydra (`${oc.env:VAR}`), and `os.environ[...]` all pick up these values.
"""
import os

# Weights and Biases
# Get your key from https://wandb.ai/authorize. 
# Leave WANDB_API_KEY empty and run `wandb login` once if you'd rather store it in ~/.netrc.
os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_ENTITY"]  = ""
os.environ["WANDB_PROJECT"] = "latentft"

# The official MTG Jamendo split-0 directory from
# https://github.com/MTG/mtg-jamendo-dataset (clone it and point here):
#   <repo>/data/splits/split-0/
os.environ["MTG_JAMENDO_SPLITS_DIR"] = "/path/to/mtg-jamendo-dataset/data/splits/split-0"

# Path to raw MTG Jamendo audio (MP3s)
os.environ["MTG_JAMENDO_RAW_DIR"] = "/path/to/mtg-jamendo/raw"

# Processed datasets. Expected layout:
#   <PROCESSED_DATA_DIR>/
#     mtg-jamendo/full-5s/   # Generated using `python latentft/data/mtg_jamendo.py full-5s`
#     gtzan/                 # Optional, only needed to reproduce sweep results. Store VGGish features + genre labels (classification)
os.environ["PROCESSED_DATA_DIR"] = "/path/to/processed-datasets"

# Training runs go to $EXP_DIR/runs/<name>/.
os.environ["EXP_DIR"] = "./exp"

# Drop rave_pretrained_musicnet.ts here, only needed to reproduce that baseline
os.environ["PRETRAINED_DIR"] = "./exp/pretrained_checkpoints"
