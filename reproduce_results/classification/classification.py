import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

import lightning as L
from lightning import seed_everything

import wandb

from fmdiffae.arc.unet1d import Conv1d, ConvBlock
from fmdiffae.training.lit_data_module import BaseDataModule


class Classifier(nn.Module):
    def __init__(
        self,
        data_resolution=512,
        in_channels=80,
        num_classes=10,
        model_dim=128,
        channel_mults=[1, 1, 1, 1, 1, 2],  # 512, 256, 128, 64, 32, 16
        num_blocks_per_res=2,
        kernel_size=3,
        use_attention=True,
        attn_resolutions=[32, 16, 8],
        num_heads=1,
    ):
        super().__init__()

        # Filling out Fields
        self.data_resolution = data_resolution
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.model_dim = model_dim
        self.channel_mults = channel_mults
        self.num_blocks_per_res = num_blocks_per_res

        self.kernel_size = kernel_size

        self.use_attention = use_attention
        self.attn_resolutions = attn_resolutions
        self.num_heads = num_heads

        self.block_kwargs = dict(
            kernel_size=kernel_size,
            num_heads=num_heads,
            use_t=False,
        )

        # Number of resolutions
        self.num_levels = len(channel_mults)
        self._build_encoder()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(channel_mults[-1] * model_dim, num_classes)

        # Print number of params
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"Classifier Number of Parameters:  {self.num_params}")

    def forward(self, x):
        for name, module in self.enc.items():
            x = module(x, None) if isinstance(module, ConvBlock) else module(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)

    def _build_encoder(self):
        self.enc = torch.nn.ModuleDict()

        for level in range(self.num_levels):
            res = self.data_resolution >> level
            res_out_channels = self.channel_mults[level] * self.model_dim

            if level == 0:
                res_in_channels = self.model_dim

                self.enc[f"{res}_conv0"] = Conv1d(
                    in_channels=self.in_channels,
                    out_channels=res_in_channels,
                    kernel_size=self.kernel_size,
                )

            else:
                res_in_channels = self.channel_mults[level - 1] * self.model_dim

                self.enc[f"{res * 2}->{res}_down"] = ConvBlock(
                    in_channels=res_in_channels,
                    out_channels=res_in_channels,
                    down=True,
                    **self.block_kwargs,
                )

            for block_idx in range(self.num_blocks_per_res):
                block_in_channels = (
                    res_in_channels if block_idx == 0 else res_out_channels
                )

                self.enc[f"{res}_block{block_idx}"] = ConvBlock(
                    in_channels=block_in_channels,
                    out_channels=res_out_channels,
                    use_attention=(res in self.attn_resolutions) and self.use_attention,
                    dilation=1,
                    **self.block_kwargs,
                )


class ClassificationDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = torch.from_numpy(np.load(data_path))
        self.labels = torch.from_numpy(np.load(labels_path))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])


class ClassifierModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        # Allows providing regular dicts without blowing up instantiate
        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)

        # Params are saved as primitives
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
        self.model = instantiate(config.model)

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.model.parameters())
        return optimizer

    def compute_loss(self, batch):
        x, y = batch
        logits = self(x)
        return nn.functional.cross_entropy_loss(logits, y)

    def training(self, batch):
        loss = self.compute_loss(self, batch)
        self.log("loss/train", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def validation_step(self, batch):
        loss = self.compute_loss(self, batch)
        self.log("loss/valid", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss


@hydra.main(
    version_base=None,
    config_path="exp/configs",
    config_name="default",
)
def main(config):
    seed_everything(config.seed, workers=True)
    wandb.login(key="2ed9110b61c4bd8c0534e383f5373cd0cc7919af")
    torch.set_float32_matmul_precision(config.float32_matmul_precision)

    data_module = BaseDataModule(config.data)
    lit_module = ClassifierModule(config)

    if config.compile:
        lit_module = torch.compile(lit_module)

    logger = instantiate(
        config.logger, resume=("auto" if config.ckpt_path else "never")
    )

    trainer = instantiate(config.trainer, logger=logger)
    trainer.fit(lit_module, data_module, ckpt_path=config.ckpt_path)


if __name__ == "__main__":
    main()
