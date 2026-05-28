import io
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torchmetrics import CohenKappa, ConfusionMatrix
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

from disney_clf.models.baseline import BaselineCNN
from disney_clf.models.main_model import EfficientNetClassifier

matplotlib.use("Agg")


def build_model(cfg: DictConfig):
    if cfg.model.name == "baseline":
        return BaselineCNN(cfg.model)
    return EfficientNetClassifier(cfg.model)


def _build_metric_set(num_classes: int) -> dict:
    return {
        "accuracy": MulticlassAccuracy(num_classes=num_classes),
        "macro_f1": MulticlassF1Score(num_classes=num_classes, average="macro"),
        "precision_macro": MulticlassPrecision(
            num_classes=num_classes, average="macro"
        ),
        "recall_macro": MulticlassRecall(num_classes=num_classes, average="macro"),
        "cohen_kappa": CohenKappa(task="multiclass", num_classes=num_classes),
    }


class DisneyClassifier(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = build_model(cfg)
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))

        num_classes = cfg.model.num_classes
        self.num_classes = num_classes

        self.train_metrics = torch.nn.ModuleDict(_build_metric_set(num_classes))
        self.val_metrics = torch.nn.ModuleDict(_build_metric_set(num_classes))
        self.test_metrics = torch.nn.ModuleDict(_build_metric_set(num_classes))

        self.val_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.test_confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)

        self._history: dict[str, list[tuple[int, float]]] = {}

    def forward(self, x):
        return self.model(x)

    def _log_metrics(self, prefix: str, metrics: torch.nn.ModuleDict, preds, y):
        for name, metric in metrics.items():
            metric(preds, y)
            self.log(
                f"{prefix}/{name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=(name in {"accuracy", "macro_f1"}),
            )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_metrics("train", self.train_metrics, preds, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self._log_metrics("val", self.val_metrics, preds, y)
        self.val_confmat.update(preds, y)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = logits.argmax(dim=1)
        self.log("test/loss", loss, on_epoch=True)
        self._log_metrics("test", self.test_metrics, preds, y)
        self.test_confmat.update(preds, y)

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
            self.logger.experiment.set_tag(self.logger.run_id, "git_commit", git_hash)
        except Exception:
            pass

    def on_train_epoch_start(self):
        epoch = self.current_epoch
        freeze_until = getattr(self.cfg.model, "freeze_backbone_epochs", 0)
        if hasattr(self.model, "features"):
            for param in self.model.features.parameters():
                param.requires_grad = epoch >= freeze_until

    def _record_history(self, prefix: str):
        for key, val in self.trainer.callback_metrics.items():
            if not key.startswith(f"{prefix}/"):
                continue
            try:
                value = float(val)
            except (TypeError, ValueError):
                continue
            history = self._history.setdefault(key, [])
            if history and history[-1][0] == self.current_epoch:
                history[-1] = (self.current_epoch, value)
            else:
                history.append((self.current_epoch, value))

    def on_train_epoch_end(self):
        self._record_history("train")

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            self.val_confmat.reset()
            return
        self._record_history("val")
        self._log_confusion_matrix("val", self.val_confmat)
        self.val_confmat.reset()

    def on_test_epoch_end(self):
        self._log_confusion_matrix("test", self.test_confmat)
        self.test_confmat.reset()

    def on_train_end(self):
        self._log_curves()

    def dump_plots(self, plots_dir: Path) -> None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)

        loss_keys = [k for k in self._history if k.endswith("/loss")]
        if loss_keys:
            self._save_curve_figure(
                loss_keys, "Loss", "loss", plots_dir / "loss_curves.png"
            )

        for name in [
            "accuracy",
            "macro_f1",
            "precision_macro",
            "recall_macro",
            "cohen_kappa",
        ]:
            keys = [k for k in self._history if k.endswith(f"/{name}")]
            if keys:
                self._save_curve_figure(
                    keys, name, name, plots_dir / f"{name}_curves.png"
                )

    def _save_curve_figure(self, keys, title, ylabel, out_path: Path) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in sorted(keys):
            points = self._history[key]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=key)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    def _mlflow_client(self):
        logger = self.logger
        if not hasattr(logger, "experiment") or not hasattr(logger, "run_id"):
            return None, None
        return logger.experiment, logger.run_id

    def _log_confusion_matrix(self, stage: str, confmat: ConfusionMatrix):
        client, run_id = self._mlflow_client()
        if client is None:
            return
        matrix = confmat.compute().detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(f"{stage} confusion matrix (epoch {self.current_epoch})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_xticks(range(self.num_classes))
        ax.set_yticks(range(self.num_classes))
        thresh = matrix.max() / 2.0 if matrix.max() > 0 else 0.5
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    int(matrix[i, j]),
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > thresh else "black",
                )
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        artifact = f"plots/{stage}_confusion_matrix_epoch_{self.current_epoch:03d}.png"
        self._log_figure(client, run_id, fig, artifact)
        plt.close(fig)

    def _log_curves(self):
        client, run_id = self._mlflow_client()
        if client is None or not self._history:
            return

        loss_keys = [k for k in self._history if k.endswith("/loss")]
        if loss_keys:
            self._log_curve_figure(
                client,
                run_id,
                loss_keys,
                title="Loss",
                ylabel="loss",
                artifact="plots/loss_curves.png",
            )

        metric_names = [
            "accuracy",
            "macro_f1",
            "precision_macro",
            "recall_macro",
            "cohen_kappa",
        ]
        for name in metric_names:
            keys = [k for k in self._history if k.endswith(f"/{name}")]
            if keys:
                self._log_curve_figure(
                    client,
                    run_id,
                    keys,
                    title=name,
                    ylabel=name,
                    artifact=f"plots/{name}_curves.png",
                )

    def _log_curve_figure(self, client, run_id, keys, title, ylabel, artifact):
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in sorted(keys):
            points = self._history[key]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", label=key)
        ax.set_xlabel("epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        self._log_figure(client, run_id, fig, artifact)
        plt.close(fig)

    def _log_figure(self, client, run_id, fig, artifact_path: str):
        try:
            client.log_figure(run_id, fig, artifact_path)
            return
        except Exception:
            pass
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        buf.seek(0)
        tmp = Path(tempfile.gettempdir()) / Path(artifact_path).name
        tmp.write_bytes(buf.getvalue())
        try:
            client.log_artifact(
                run_id, str(tmp), artifact_path=str(Path(artifact_path).parent)
            )
        except Exception:
            pass
