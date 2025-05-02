from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import lightning as L
from hydra.utils import instantiate


class FMDiffAEModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        self.model = instantiate(config.model)
        self.transform = instantiate(config.data.transform)

        if config.use_ema_weights:
            self.ema_model = AveragedModel(
                self.model, multi_avg_fn=get_ema_multi_avg_fn(config.ema_decay)
            )

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def training_step(self, batch):
        loss = self(batch)
        self.log(
            "loss/train",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, "ema_model"):
            self.ema_model.update_parameters(self.model)

    def validation_step(self, batch):
        loss = self(batch)
        self.log(
            "loss/valid",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.model.parameters())
        scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_fit_start(self):
        self.transform.model.to(self.device)

    def load_torch_model(cls, ckpt_path):
        lit = cls.load_from_checkpoint(ckpt_path)
        return lit.ema_model.module if hasattr(lit, "ema_model") else lit.model
