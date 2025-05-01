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

    data_module = BaseDataModule(config.data)
    lit_module = FMDiffAEModule(config)

    if config.compile:
        lit_module = torch.compile(lit_module)

    callbacks = [instantiate(c) for c in config.callbacks.values()]

    logger = instantiate(config.logger)
    trainer = instantiate(config.trainer, callbacks=callbacks, logger=logger)

    trainer.fit(lit_module, data_module, ckpt_path=config.ckpt_path)


if __name__ == "__main__":
    main()
