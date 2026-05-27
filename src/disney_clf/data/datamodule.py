import tempfile
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from disney_clf.data.dataset import DisneyDataset
from disney_clf.data.preprocess import build_manifest


def download_data(
    raw_dir: str, processed_dir: str, remote: str = "data_storage"
) -> None:
    processed = Path(processed_dir)
    if (processed / "train.csv").exists() and (processed / "test.csv").exists():
        return

    try:
        from dvc.repo import Repo

        with Repo() as repo:
            repo.pull(remote=remote)
        if (processed / "train.csv").exists() and (processed / "test.csv").exists():
            return
    except Exception as dvc_exc:
        print(f"[data] dvc pull failed ({dvc_exc}); falling back to Kaggle API")

    try:
        import kaggle

        raw = Path(raw_dir)
        raw.mkdir(parents=True, exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "sayehkargari/disney-characters-dataset", path=str(raw), unzip=True
        )
        build_manifest(str(raw), str(processed))
    except Exception as kaggle_exc:
        raise RuntimeError(
            f"Failed to fetch data via DVC and Kaggle API ({kaggle_exc}). "
            "Either run `dvc pull -r data_storage` with access to the remote, "
            "or place `~/.kaggle/kaggle.json` and rerun."
        ) from kaggle_exc


class DisneyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def prepare_data(self) -> None:
        download_data(self.cfg.raw_dir, self.cfg.processed_dir)

    def _train_transforms(self):
        aug = self.cfg.augmentation
        return transforms.Compose(
            [
                transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
                (
                    transforms.RandomHorizontalFlip()
                    if aug.random_horizontal_flip
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.RandomRotation(aug.random_rotation),
                transforms.ColorJitter(
                    brightness=aug.color_jitter.brightness,
                    contrast=aug.color_jitter.contrast,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def _eval_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize((self.cfg.image_size, self.cfg.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def setup(self, stage=None):
        train_df = pd.read_csv(self.cfg.train_csv)
        train_split, val_split = train_test_split(
            train_df, test_size=0.1, stratify=train_df["label"], random_state=42
        )
        tmp_dir = Path(tempfile.gettempdir())
        train_split_path = tmp_dir / "disney_train_split.csv"
        val_split_path = tmp_dir / "disney_val_split.csv"
        train_split.to_csv(train_split_path, index=False)
        val_split.to_csv(val_split_path, index=False)

        self.train_dataset = DisneyDataset(
            str(train_split_path),
            transform=self._train_transforms(),
        )
        self.val_dataset = DisneyDataset(
            str(val_split_path),
            transform=self._eval_transforms(),
        )
        self.test_dataset = DisneyDataset(
            self.cfg.test_csv, transform=self._eval_transforms()
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            shuffle=False,
        )
