import torch.nn as nn
from omegaconf import DictConfig
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetClassifier(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if cfg.pretrained else None
        backbone = efficientnet_b0(weights=weights)

        self.upsample = nn.Upsample(
            size=(224, 224), mode="bilinear", align_corners=False
        )
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        head_cfg = cfg.head
        self.classifier = nn.Sequential(
            nn.Dropout(p=head_cfg.dropout1),
            nn.Linear(1280, head_cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=head_cfg.dropout2),
            nn.Linear(head_cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, x):
        x = self.upsample(x)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        return self.classifier(x)
