import lightning as L
from torch.utils.data import DataLoader
from hydra.utils import instantiate


class BaseDataModule(L.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.save_hyperparameters(data_config)

        self.train_ds = None
        self.valid_ds = None
        self.valid_vggish_embeddings = None
        self.sample_rate = data_config.sample_rate
        self.batch_size = data_config.batch_size

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_ds = instantiate(self.hparams.dataset, split="train")

        if stage in (None, "fit", "validate"):
            self.valid_ds = instantiate(self.hparams.dataset, split="valid")

            self.valid_vggish_embeddings = getattr(
                self.valid_ds, "valid_vggish_embeddings", None
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
