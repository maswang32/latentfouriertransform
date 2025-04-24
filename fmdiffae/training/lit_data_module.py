import lightning as L
from torch.utils.data import DataLoader


class BaseDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_cls,
        dataset_kwargs,
        batch_size=256,
        num_workers=4,
        pin_memory=False,
    ):
        super().__init__
        self.dataset_cls = dataset_cls
        self.dataset_kwargs = dataset_kwargs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        self.train_ds = self.dataset_cls(split="train", **self.dataset_kwargs)
        self.valid_ds = self.dataset_cls(split="valid", **self.dataset_kwargs)
        self.valid_vggish_embeddings = getattr(
            self.valid_ds, "valid_vggish_embeddings", None
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def validation_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
