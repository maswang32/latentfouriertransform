import lightning as L
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn


class FMDiffAEModule(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        # Allows providing regular dicts without blowing up instantiate
        if not isinstance(config, DictConfig):
            config = OmegaConf.create(config)

        # Params are saved as primitives
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
        self.strict_loading = config.strict_loading
        self.model = instantiate(config.model)
        self.transform = instantiate(config.data.transform)

        if self.hparams.use_ema_weights:
            self.ema_model = AveragedModel(
                self.model, multi_avg_fn=get_ema_multi_avg_fn(self.hparams.ema_decay)
            )
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()

    def on_fit_start(self):
        self.transform.model.to(self.device)

    def configure_optimizers(self):
        optimizer = instantiate(self.hparams.optimizer, params=self.model.parameters())
        scheduler = instantiate(self.hparams.scheduler, optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_train_start(self):
        target_lr = self.hparams["optimizer"]["lr"]
        for optimizer in self.trainer.optimizers:
            for param_group in optimizer.param_groups:
                old_lr = param_group["lr"]
                param_group["lr"] = target_lr
                print(f"Changed LR from {old_lr} to {param_group['lr']}")

    def training_step(self, batch):
        loss = self(batch)
        self.log("loss/train", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.model(x, *args, **kwargs)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, "ema_model"):
            self.ema_model.update_parameters(self.model)

    def validation_step(self, batch):
        loss = self(batch)
        self.log("loss/valid", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    @classmethod
    def load_torch_model(cls, ckpt_path, strict=True):
        lit = cls.load_from_checkpoint(ckpt_path, strict=strict)
        print("Loading EMA Model?", hasattr(lit, "ema_model"))
        return lit.ema_model.module if hasattr(lit, "ema_model") else lit.model
