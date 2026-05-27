import onnxruntime as ort
import torch
from omegaconf import DictConfig

from disney_clf.training.module import DisneyClassifier


def export_to_onnx(cfg: DictConfig) -> None:
    module = DisneyClassifier.load_from_checkpoint(
        cfg.inference.checkpoint_path, cfg=cfg
    )
    module.eval()

    dummy = torch.randn(1, 3, cfg.data.image_size, cfg.data.image_size)

    torch.onnx.export(
        module.model,
        dummy,
        cfg.inference.onnx_path,
        opset_version=cfg.inference.opset_version,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )
    print(f"Exported ONNX model to {cfg.inference.onnx_path}")

    sess = ort.InferenceSession(cfg.inference.onnx_path)
    out = sess.run(None, {"image": dummy.numpy()})
    assert out[0].shape == (1, cfg.model.num_classes), "ONNX output shape mismatch"
    print("ONNX verification passed.")
