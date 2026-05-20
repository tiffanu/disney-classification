import hydra
from omegaconf import DictConfig

from disney_clf.inference.export_onnx import export_to_onnx


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    export_to_onnx(cfg)


if __name__ == "__main__":
    main()
