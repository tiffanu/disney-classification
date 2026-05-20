from typing import Dict, List

import numpy as np
import onnxruntime as ort
from PIL import Image

CLASSES = [
    "donald_duck",
    "mickey_mouse",
    "minion",
    "olaf",
    "winnie_the_pooh",
    "pumba",
]


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _preprocess(image_path: str, image_size: int = 64) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return arr.transpose(2, 0, 1)[np.newaxis]


def predict(
    image_path: str,
    model_path: str,
    classes: List[str] = CLASSES,
    image_size: int = 64,
) -> Dict:
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    img = _preprocess(image_path, image_size)
    logits = sess.run(None, {"image": img})[0]
    probs = _softmax(logits)[0]
    top_idx = int(probs.argmax())
    return {
        "class": classes[top_idx],
        "confidence": float(probs[top_idx]),
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
    }
