import urllib.parse
import urllib.request
from pathlib import Path

import hydra
import pytorch_lightning as pl
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import MLFlowLogger

from disney_clf.data.datamodule import DisneyDataModule
from disney_clf.training.module import DisneyClassifier


def _resolve_tracking_uri(cfg: DictConfig, root: Path) -> str:
    configured = cfg.mlflow.tracking_uri
    parsed = urllib.parse.urlparse(configured)
    if parsed.scheme in {"http", "https"}:
        try:
            urllib.request.urlopen(configured, timeout=1.5)
            return configured
        except Exception:
            fallback = root / "mlruns"
            print(
                f"[mlflow] server {configured} unreachable, "
                f"falling back to file:{fallback}"
            )
            return f"file:{fallback}"
    return configured


def main(cfg: DictConfig) -> None:
    root = Path(get_original_cwd())
    tracking_uri = _resolve_tracking_uri(cfg, root)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.experiment_name,
        run_name=cfg.model.name,
        tracking_uri=tracking_uri,
        log_model=True,
    )
    mlflow_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    datamodule = DisneyDataModule(cfg.data)
    module = DisneyClassifier(cfg)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(root / "models"),
            monitor=cfg.training.checkpoint.monitor,
            mode=cfg.training.checkpoint.mode,
            save_top_k=cfg.training.checkpoint.save_top_k,
            filename=cfg.training.checkpoint.filename,
        ),
        EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        gradient_clip_val=cfg.training.gradient_clip_val,
        callbacks=callbacks,
        logger=mlflow_logger,
        log_every_n_steps=10,
    )

    trainer.fit(module, datamodule)
    trainer.test(module, datamodule, ckpt_path="best")

    plots_dir = root / "plots"
    module.dump_plots(plots_dir)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def _entrypoint(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    _entrypoint()
