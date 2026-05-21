import numpy as np
import pandas as pd
import pytest
from PIL import Image
from torchvision import transforms

from disney_clf.data.dataset import DisneyDataset


@pytest.fixture
def sample_dataset(tmp_path):
    records = []
    for i in range(6):
        img = Image.fromarray(
            (np.random.rand(64, 64, 3) * 255).astype(np.uint8), mode="RGB"
        )
        img_path = tmp_path / f"img_{i}.jpg"
        img.save(img_path)
        records.append({"path": str(img_path), "label": i % 6})

    csv_path = tmp_path / "data.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return str(csv_path)


def test_dataset_len(sample_dataset):
    ds = DisneyDataset(sample_dataset)
    assert len(ds) == 6


def test_dataset_item_shape(sample_dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    ds = DisneyDataset(sample_dataset, transform=transform)
    img, label = ds[0]
    assert img.shape == (3, 64, 64)
    assert isinstance(label, int)
