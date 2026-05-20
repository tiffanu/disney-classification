import subprocess

import mlflow
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torchmetrics import CohenKappa
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from disney_clf.models.baseline import BaselineCNN
from disney_clf.models.main_model import EfficientNetClassifier


def build_model(cfg: DictConfig):
    if cfg.model.name == "baseline":
        return BaselineCNN(cfg.model)
    return EfficientNetClassifier(cfg.model)


class DisneyClassifier(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        num_classes = cfg.model.num_classes
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_precision = MulticlassPrecision(
            num_classes=num_classes, average="macro"
        )
        self.val_recall = MulticlassRecall(num_classes=num_classes, average="macro")
        self.val_kappa = CohenKappa(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/accuracy", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)
        self.val_kappa(preds, y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/accuracy", self.val_acc, on_epoch=True)
        self.log("val/macro_f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val/precision_macro", self.val_precision, on_epoch=True)
        self.log("val/recall_macro", self.val_recall, on_epoch=True)
        self.log("val/cohen_kappa", self.val_kappa, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.log("test/loss", loss)
        n = self.cfg.model.num_classes
        self.log(
            "test/accuracy",
            MulticlassAccuracy(num_classes=n).to(self.device)(preds, y),
        )
        self.log(
            "test/macro_f1",
            MulticlassF1Score(num_classes=n, average="macro").to(self.device)(preds, y),
        )

    def configure_optimizers(self):
        opt_cfg = self.cfg.training.optimizer
        sched_cfg = self.cfg.training.scheduler

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.T_max,
            eta_min=sched_cfg.eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_train_start(self):
        try:
            git_hash = (
                subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            )
            mlflow.set_tag("git_commit", git_hash)
        except Exception:
            pass

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        freeze_until = getattr(self.cfg.model, "freeze_backbone_epochs", 0)
        if hasattr(self.model, "features"):
            for param in self.model.features.parameters():
                param.requires_grad = epoch >= freeze_until
