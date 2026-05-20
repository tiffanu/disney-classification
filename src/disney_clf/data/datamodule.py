import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from disney_clf.data.dataset import DisneyDataset


class DisneyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

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
        import pandas as pd

        train_df = pd.read_csv(self.cfg.train_csv)
        train_split, val_split = train_test_split(
            train_df, test_size=0.1, stratify=train_df["label"], random_state=42
        )
        train_split.to_csv("/tmp/disney_train_split.csv", index=False)
        val_split.to_csv("/tmp/disney_val_split.csv", index=False)

        self.train_dataset = DisneyDataset(
            "/tmp/disney_train_split.csv",
            transform=self._train_transforms(),
        )
        self.val_dataset = DisneyDataset(
            "/tmp/disney_val_split.csv",
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
