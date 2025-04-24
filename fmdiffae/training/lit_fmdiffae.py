import lightning as L
from fmdiffae.diffusion.fmdiffae import FMDiffAE
from hydra.utils import instantiate


class FMDiffAEModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.encoder = cfg.encoder_cls(**cfg.encoder_kwargs)
        self.decoder = cfg.decoder_cls(**cfg.decoder_kwargs)

        self.model = FMDiffAE(encoder=self.encoder, decoder=self.decoder)
        self.transform = cfg.transform_class(**cfg.transform_kwargs)

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self(batch)
        self.log("valid/loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer_
