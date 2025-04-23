import os
import time
import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml
from typing import Type, Dict, Any
from dataclasses import dataclass, asdict, field
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from fmdiffae.data.voice import VoiceSpectrogramDataset


@dataclass
class Config:
    # Save Settings
    save_name: str
    save_base_dir: str = "exp"
    load_checkpoint_if_avail: bool = True

    # Data settings
    dataset_class: Type[Dataset] = VoiceSpectrogramDataset
    dataset_kwargs: Dict[str, Any] = field(default_factory=dict)
    num_workers: int = 4
    pin_memory: bool = False

    # Optimizer Settings
    optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    clip_grad_norm: float | None = 1.0

    # Training Settings
    batch_size: int = 256
    device: torch.device = torch.device("cuda:0")

    num_train_iters: int = 10_000_000
    num_warmup_iters: int = 4000
    num_iters_per_eval: int = 570
    num_valid_iters: int = 4

    use_ema_weights: bool = True
    ema_decay: float = 0.999
    ema_warmup_iters: int = 4000


class Trainer:
    """
    Notes:
        - Assume the dataset class has a split parameter,
            with "train" and "valid" as options.
        - Assume that the model's forward function gives the loss.
    """

    def __init__(self, config, model):
        # Assign fields
        self.config = config
        self.model = model

        self.init_dataloaders()
        self.init_optimizer_and_scheduler()
        self.init_save_directories()
        self.init_logging()

        if self.config.use_ema_weights:
            self.init_ema_model()

        # Load Model/Losses if continuing training
        if config.load_checkpoint_if_avail and os.path.exists(self.checkpoint_path):
            self.load_checkpoint()

    def init_dataloaders(self):
        # Datasets
        self.dataset_train = self.config.dataset_class(
            split="train", **self.config.dataset_kwargs
        )
        self.dataset_valid = self.config.dataset_class(
            split="valid", **self.config.dataset_kwargs
        )

        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        self.dataloader_valid = DataLoader(
            self.dataset_valid,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

    def init_optimizer_and_scheduler(self):
        # Optimizer
        self.optimizer = self.config.optimizer_class(
            params=self.model.parameters(), **self.config.optimizer_kwargs
        )

        # Scheduler
        self.scheduler = None
        if self.config.num_warmup_iters > 0:

            def warmup_lr_schedule(step: int):
                factor = (step + 1) / self.config.num_warmup_iters
                return min(factor, 1.0)

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=warmup_lr_schedule,
                last_epoch=-1,
            )

    def init_save_directories(self):
        # Make Save dir, and place to save examples
        self.save_dir = os.path.join(self.config.save_base_dir, self.config.save_name)
        os.makedirs(self.save_dir, exist_ok=True)

        # Paths to checkpoint, train losses, valid losses
        self.checkpoint_path = os.path.join(self.save_dir, "checkpoint.pt")
        self.train_losses_path = os.path.join(self.save_dir, "train_losses.npy")
        self.valid_losses_path = os.path.join(self.save_dir, "valid_losses.npy")
        self.config_path = os.path.join(self.save_dir, "config.yml")

    def init_logging(self):
        # Tracking
        self.step = 0
        self.num_epochs_seen = 0
        self.num_datapoints_seen = 0
        self.total_training_time = 0.0
        self.train_losses = []
        self.valid_losses = []

    def init_ema_model(self):
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.load_state_dict(self.model.state_dict())
        self.ema_model.requires_grad_(False)
        self.current_ema_decay = 0.0

    def save_checkpoint(self):
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "ema_model_state_dict": self.ema_model.state_dict()
            if self.config.use_ema_weights
            else None,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict()
            if self.scheduler
            else None,
            "step": self.step,
            "num_epochs_seen": self.num_epochs_seen,
            "num_datapoints_seen": self.num_datapoints_seen,
            "total_training_time": self.total_training_time,
        }
        torch.save(checkpoint, self.checkpoint_path)

        np.save(self.train_losses_path, self.train_losses)
        np.save(self.valid_losses_path, self.valid_losses)

        with open(self.config_path, "w") as f:
            yaml.dump(asdict(self.config), f, default_flow_style=False)

    def load_checkpoint(self):
        tqdm.write(f"Loading from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.config.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])

        if self.config.use_ema_weights:
            if checkpoint.get("ema_model_state_dict") is not None:
                self.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
            else:
                self.ema_model.load_state_dict(self.model.state_dict())

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler:
            if checkpoint["scheduler_state_dict"]:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            else:
                raise NotImplementedError(
                    "Resuming scheduler without checkpoint not implemented"
                )

        self.step = checkpoint["step"]
        self.num_epochs_seen = checkpoint["num_epochs_seen"]
        self.num_datapoints_seen = checkpoint["num_datapoints_seen"]
        self.total_training_time = checkpoint["total_training_time"]

        if os.path.exists(self.train_losses_path):
            self.train_losses = list(np.load(self.train_losses_path))

        if os.path.exists(self.valid_losses_path):
            self.valid_losses = list(np.load(self.valid_losses_path))

    @torch.no_grad()
    def _update_ema(self):
        if self.step < self.config.ema_warmup_iters:
            self.current_ema_decay = (
                self.step / self.config.ema_warmup_iters
            ) * self.config.ema_decay
        else:
            self.current_ema_decay = self.config.ema_decay

        ema_params = list(self.ema_model.parameters())
        model_params = list(self.model.parameters())

        torch._foreach_mul_(ema_params, self.current_ema_decay)
        torch._foreach_add_(ema_params, model_params, alpha=1 - self.current_ema_decay)

        for ema_b, model_b in zip(self.ema_model.buffers(), self.model.buffers()):
            ema_b.copy_(model_b)

    def train_step(self, batch):
        # Training Step
        self.optimizer.zero_grad()

        # Forward/Backward passes
        batch = batch.to(self.config.device)
        loss = self.model(batch)
        loss.backward()

        # Gradient Clipping
        if self.config.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.config.clip_grad_norm
            )

        # Parameter and scheduler update
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if self.config.use_ema_weights:
            self._update_ema()

        # Track Losses
        self.train_losses.append([self.step, loss.item()])

        # Increment Statistics
        self.num_datapoints_seen += len(batch)
        self.step += 1

    @torch.no_grad()
    def compute_valid_loss(self):
        if self.config.num_valid_iters == 0:
            return 0.0

        running_loss = 0.0

        model = self.ema_model if self.config.use_ema_weights else self.model

        for i, batch in enumerate(
            tqdm(
                self.dataloader_valid,
                total=self.config.num_valid_iters,
                desc="Evaluating",
                leave=False,
            )
        ):
            if i >= self.config.num_valid_iters:
                break
            batch = batch.to(self.config.device)
            running_loss += model(batch).item()

        return running_loss / min(
            len(self.dataloader_valid), self.config.num_valid_iters
        )

    @torch.no_grad()
    def eval_cycle(self):
        # Change model to eval mode
        self.model.eval()

        # Compute Valid Losses
        valid_loss = self.compute_valid_loss()
        tqdm.write(f"Valid Loss: {valid_loss:.3f}")
        self.valid_losses.append(
            [
                self.step,
                valid_loss,
                self.num_epochs_seen,
                self.total_training_time,
                self.num_datapoints_seen,
            ]
        )

        # Save Checkpoint
        tqdm.write("Saving Checkpoint...")
        self.save_checkpoint()

    def train(self):
        start_time = time.time()

        while self.step < self.config.num_train_iters:
            pbar = tqdm(
                self.dataloader_train,
                total=len(self.dataloader_train),
                desc=f"Epoch {self.num_epochs_seen + 1}",
                leave=False,
            )

            for batch in pbar:
                self.train_step(batch)

                if self.step % 10 == 0:
                    avg_loss = np.mean(self.train_losses[-10:], axis=0)[1]
                    pbar.set_postfix(
                        avg10=f"{avg_loss:.3f}",
                    )

                # Evaluation Loop
                if self.step % self.config.num_iters_per_eval == 0:
                    # Track total training time
                    self.total_training_time += time.time() - start_time
                    tqdm.write(f"Train Time: {(self.total_training_time / 60):.1f} min")

                    self.eval_cycle()

                    # Change model to train mode
                    self.model.train()

                    # Start Timer Again
                    start_time = time.time()

                if self.step >= self.config.num_train_iters:
                    break

            if self.step < self.config.num_train_iters:
                self.num_epochs_seen += 1

    def save_loss_curves(self):
        train_losses = np.array(self.train_losses)
        valid_losses = np.array(self.valid_losses)

        plt.plot(train_losses[:, 0], train_losses[:, 1], label="train")
        plt.plot(valid_losses[:, 0], valid_losses[:, 1], label="valid")
        plt.title(
            f"Losses at step={self.step}",
        )
        plt.xlabel("Step")
        plt.ylabel("Train/Valid Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "losses.png"))
        plt.close()
