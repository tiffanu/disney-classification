import hydra
from omegaconf import DictConfig

from disney_clf.data.preprocess import build_manifest


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    build_manifest(
        raw_dir=cfg.data.raw_dir,
        processed_dir=cfg.data.processed_dir,
    )


if __name__ == "__main__":
    main()
