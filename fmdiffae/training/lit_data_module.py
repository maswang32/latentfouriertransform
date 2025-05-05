import lightning as L
from hydra.utils import instantiate


class BaseDataModule(L.LightningDataModule):
    def __init__(self, data_config):
        super().__init__()
        self.save_hyperparameters(data_config)
        self.train_ds = None
        self.valid_ds = None
        self.sample_rate = data_config.sample_rate
        self.batch_size = data_config.batch_size

    def setup(self, stage=None):
        if stage in (None, "fit"):
            self.train_ds = instantiate(self.hparams.train_dataset)

        if stage in (None, "fit", "validate"):
            self.valid_ds = instantiate(self.hparams.valid_dataset)

    def train_dataloader(self):
        return instantiate(self.hparams.train_dataloader, dataset=self.train_ds)

    def val_dataloader(self):
        return instantiate(self.hparams.valid_dataloader, dataset=self.valid_ds)
