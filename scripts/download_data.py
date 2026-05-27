from pathlib import Path

import hydra
import kaggle
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset_slug = "sayehkargari/disney-characters-dataset"
    output_dir = Path(cfg.data.raw_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_slug, path=str(output_dir), unzip=True)
    print(f"Dataset downloaded to {output_dir}")


if __name__ == "__main__":
    main()
