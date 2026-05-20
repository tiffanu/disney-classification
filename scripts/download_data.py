import os

import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    import kaggle

    dataset_slug = "sayehkargari/disney-characters-dataset"
    output_dir = cfg.data.raw_dir

    os.makedirs(output_dir, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(dataset_slug, path=output_dir, unzip=True)
    print(f"Dataset downloaded to {output_dir}")


if __name__ == "__main__":
    main()
