import hydra
import mlflow
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from disney_clf.data.datamodule import DisneyDataModule
from disney_clf.training.module import DisneyClassifier


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(cfg.experiment_name)

    with mlflow.start_run():
        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))

        datamodule = DisneyDataModule(cfg.data)
        module = DisneyClassifier(cfg)

        callbacks = [
            ModelCheckpoint(
                dirpath="models/",
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
            log_every_n_steps=10,
        )

        trainer.fit(module, datamodule)
        trainer.test(module, datamodule, ckpt_path="best")

        mlflow.pytorch.log_model(module, "model")


if __name__ == "__main__":
    main()
