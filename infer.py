import json

import hydra
from omegaconf import DictConfig

from disney_clf.inference.infer import predict


def main(cfg: DictConfig) -> None:
    image_path = cfg.inference.image_path
    if image_path is None:
        raise ValueError(
            "Provide image via CLI override: "
            "python infer.py inference.image_path=path/to/image.jpg"
        )
    result = predict(
        image_path=image_path,
        model_path=cfg.inference.onnx_path,
        image_size=cfg.data.image_size,
        classes=list(cfg.data.classes),
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def _entrypoint(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    _entrypoint()
