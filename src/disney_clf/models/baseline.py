import torch.nn as nn
from omegaconf import DictConfig


class BaselineCNN(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        channels = cfg.architecture.conv_channels
        num_classes = cfg.num_classes

        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        flat_dim = channels[2] * 8 * 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, cfg.architecture.fc_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.architecture.dropout),
            nn.Linear(cfg.architecture.fc_hidden, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
