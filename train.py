import os
import torch
import hydra
import wandb
from hydra.utils import instantiate
from fmdiffae.training.lit_fmdiffae import FMDiffAEModule
from fmdiffae.training.lit_data_module import BaseDataModule


@hydra.main(
    version_base=None,
    config_path="exp/configs",
    config_name="default",
)
def main(config):
    wandb.login(key="2ed9110b61c4bd8c0534e383f5373cd0cc7919af")

    torch.set_float32_matmul_precision(config.float32_matmul_precision)

    # Finding where we should load the checkpoint
    if config.ckpt_path is not None:
        ckpt_path = config.ckpt_path
        print(f"Resuming from provided checkpoint: {ckpt_path}")
    elif config.load_last_if_avail and os.path.exists("checkpoints/last.ckpt"):
        ckpt_path = "checkpoints/last.ckpt"
        print(f"Resuming from last checkpoint: {ckpt_path}")
    else:
        ckpt_path = None
        print("No checkpoint provided or found. Training from scratch.")

    data_module = BaseDataModule(config.data)
    lit_module = FMDiffAEModule(config)

    if config.compile:
        lit_module = torch.compile(lit_module)

    callbacks = [instantiate(c) for c in config.callbacks.values()]
    logger = instantiate(config.logger, resume=("auto" if ckpt_path else "never"))
    trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(lit_module, data_module, ckpt_path=ckpt_path)


if __name__ == "__main__":
    main()
