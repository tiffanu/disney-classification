# Инференс

## Зависимости

```bash
uv sync
# или: pip install onnxruntime pillow numpy
```

## 1. Экспорт модели в ONNX

После обучения экспортируем лучший чекпойнт:

```bash
python scripts/export.py inference.checkpoint_path=models/best_main.ckpt inference.onnx_path=models/model_main.onnx
```

Скрипт автоматически проверяет форму выхода ONNX-модели.

## 2. Standalone-инференс (ONNX Runtime, без PyTorch)

```python
from src.disney_clf.inference.infer import predict

result = predict("path/to/image.jpg", "models/model_main.onnx")
# {"class": "mickey_mouse", "confidence": 0.94, "probabilities": {...}}
print(result)
```

## Требования к ресурсам

| Ресурс                       | Значение                                       |
| ---------------------------- | ---------------------------------------------- |
| CPU (инференс)               | 2+ ядра                                        |
| RAM                          | 512 MB                                         |
| GPU                          | Опционально (CPU-only ONNX Runtime достаточно) |
| Размер ONNX-модели           | ~20 MB (EfficientNet-B0)                       |
| Latency (CPU)                | ~50 мс/изображение                             |
| Latency (GPU)                | ~5 мс/изображение                              |
| Throughput (CPU)             | ~20 img/s                                      |
| Throughput (GPU, batch=32)   | ~200 img/s                                     |

## Формат входа

- Изображение: любой RGB-файл (JPEG, PNG)
- Внутренняя предобработка: resize до 64×64, нормализация mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)
- Имя ONNX-входа: `image`, форма: `(batch_size, 3, 64, 64)`, dtype: `float32`
- Имя ONNX-выхода: `logits`, форма: `(batch_size, 6)`

## Классы (индекс выхода → название)

| Индекс | Класс           |
| ------ | --------------- |
| 0      | donald_duck     |
| 1      | mickey_mouse    |
| 2      | minion          |
| 3      | olaf            |
| 4      | winnie_the_pooh |
| 5      | pumba           |
