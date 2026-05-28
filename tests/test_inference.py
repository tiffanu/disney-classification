import numpy as np
import pytest


def test_softmax_sums_to_one():
    from disney_clf.inference.infer import _softmax

    logits = np.array([[1.0, 2.0, 3.0, 0.5, -1.0, 0.0]])
    probs = _softmax(logits)
    assert abs(probs.sum() - 1.0) < 1e-6
    assert (probs >= 0).all()


def test_preprocess_output_shape():
    import tempfile

    from PIL import Image

    from disney_clf.inference.infer import _preprocess

    with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
        arr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img.save(f.name)
        arr = _preprocess(f.name, image_size=64)
    assert arr.shape == (1, 3, 64, 64)
    assert arr.dtype == np.float32


@pytest.mark.skipif(
    not __import__("os").path.exists("models/model_main.onnx"),
    reason="ONNX model not present",
)
def test_onnx_inference():
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(
        "models/model_main.onnx", providers=["CPUExecutionProvider"]
    )
    dummy = np.random.randn(1, 3, 64, 64).astype(np.float32)
    out = sess.run(None, {"image": dummy})
    assert out[0].shape == (1, 6)
