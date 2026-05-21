import torch
from omegaconf import OmegaConf

from disney_clf.models.baseline import BaselineCNN
from disney_clf.models.main_model import EfficientNetClassifier

BASELINE_CFG = OmegaConf.create(
    {
        "name": "baseline",
        "architecture": {
            "conv_channels": [32, 64, 128],
            "fc_hidden": 512,
            "dropout": 0.3,
        },
        "num_classes": 6,
    }
)

MAIN_CFG = OmegaConf.create(
    {
        "name": "efficientnet_b0",
        "backbone": "efficientnet_b0",
        "pretrained": False,
        "head": {"hidden_dim": 256, "dropout1": 0.3, "dropout2": 0.2},
        "num_classes": 6,
        "freeze_backbone_epochs": 3,
    }
)


def test_baseline_forward():
    model = BaselineCNN(BASELINE_CFG)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 6)


def test_main_model_forward():
    model = EfficientNetClassifier(MAIN_CFG)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 6)
