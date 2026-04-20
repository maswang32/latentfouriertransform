import config  # noqa: F401  -- populates os.environ with user settings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

import lightning as L
from lightning import seed_everything
from torchmetrics import Accuracy

from fmdiffae.lightning.lit_data_module import BaseDataModule


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

        self.train_accuracy = Accuracy(
            task="multiclass", num_classes=config.data.num_classes
        )

        self.valid_accuracy = Accuracy(
            task="multiclass", num_classes=config.data.num_classes
        )

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.model.parameters())
        return optimizer

    def compute_loss_and_preds(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch):
        loss, preds, y = self.compute_loss_and_preds(batch)
        self.train_accuracy(preds, y)
        self.log("loss/train", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log(
            "accuracy/train",
            self.train_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def validation_step(self, batch):
        loss, preds, y = self.compute_loss_and_preds(batch)
        self.valid_accuracy(preds, y)
        self.log("loss/valid", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "accuracy/valid",
            self.valid_accuracy,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    @classmethod
    def load_torch_model(cls, ckpt_path, strict=True):
        lit = cls.load_from_checkpoint(ckpt_path, strict=strict)
        return lit.model


@hydra.main(
    version_base=None,
    config_path="exp/configs",
    config_name="default",
)
def main(config):
    seed_everything(config.seed, workers=True)
    torch.set_float32_matmul_precision(config.float32_matmul_precision)

    data_module = BaseDataModule(config.data)
    lit_module = ClassifierModule(config)

    if config.compile:
        lit_module = torch.compile(lit_module)

    callbacks = [instantiate(c) for c in config.callbacks.values()]
    logger = instantiate(
        config.logger, resume=("auto" if config.ckpt_path else "never")
    )
    trainer = instantiate(config.trainer, logger=logger, callbacks=callbacks)
    trainer.fit(lit_module, datamodule=data_module, ckpt_path=config.ckpt_path)


if __name__ == "__main__":
    main()
