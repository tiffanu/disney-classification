# Disney Character Classification

Многоклассовая классификация 6 персонажей Disney по изображениям. PyTorch
Lightning + Hydra + MLflow + DVC + ONNX.

## Описание проекта

### Задача

Определить, какой из 6 персонажей изображён на картинке:
`donald_duck`, `mickey_mouse`, `minion`, `olaf`, `winnie_the_pooh`, `pumba`.

### Данные

Источник: [Kaggle — Disney Characters Dataset](https://www.kaggle.com/datasets/sayehkargari/disney-characters-dataset).

- 4 667 JPEG-изображений, заранее разбитых на train/test.
- 6 классов, классы несбалансированы умеренно.
- Размер ≈ 120 МБ, изображения приводятся к 64×64 при обучении.
- Хранятся вне git: raw — через `scripts/download_data.py` (Kaggle API),
  обработанные манифесты (`train.csv`, `test.csv`) — под DVC.

### Модели

В проекте две модели: лёгкий baseline для проверки пайплайна и основная — на
предобученном EfficientNet-B0.

#### Baseline CNN

3-блочная свёрточная сеть.

```
Input (3×64×64)
→ Conv(3→32, 3×3) → BN → ReLU → MaxPool(2)    # 32×32
→ Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)   # 16×16
→ Conv(64→128, 3×3) → BN → ReLU → MaxPool(2)  # 8×8
→ Flatten → Linear(8192→512) → ReLU → Dropout(0.3)
→ Linear(512→6)
```

#### EfficientNet-B0

Предобученный
`efficientnet_b0`
(веса `EfficientNet_B0_Weights.DEFAULT` из `torchvision`)

```
Input (3×64×64)
→ Upsample(224×224)
→ EfficientNet-B0 backbone (заморожен первые 3 эпохи)
→ AvgPool → Flatten(1280)
→ Dropout(0.3) → Linear(1280→256) → ReLU → Dropout(0.2)
→ Linear(256→6)
```

Ожидаемое качество на валидации:

| Модель | val/macro_f1 |
|---|---|
| Baseline CNN | ~0.72 |
| EfficientNet-B0 | ~0.95 |

### Метрики

Все считаются через [`torchmetrics`](https://lightning.ai/docs/torchmetrics/stable/):

- `accuracy` — общая доля правильных ответов
- `macro_f1` — основная метрика отбора чекпойнтов
- `precision_macro`, `recall_macro`
- `cohen_kappa`

Плюс `loss` на train/val/test.

---

## Setup

Зависимости управляются через [`uv`](https://docs.astral.sh/uv/) и описаны
в `pyproject.toml` (lock — `uv.lock`).

```bash
git clone <repo-url> disney-clf
cd disney-clf

uv sync --extra dev
source .venv/bin/activate

pre-commit install
```

## Data

Данные не лежат в git. Доступны два пути:

1. **Скачать данные с Kaggle** (нужен `~/.kaggle/kaggle.json`):

   ```bash
   python scripts/download_data.py
   python scripts/preprocess.py
   ```

2. **Скачать готовые манифесты через DVC** (если есть доступ к remote):

   ```bash
   dvc pull -r data_storage
   ```

При запуске `train.py` `DisneyDataModule.prepare_data()` сам вызывает
`dvc pull` если `data/processed/*.csv` отсутствуют.

DVC настроен на два хранилища (`.dvc/config`):

- `data_storage` — данные (`data/`)
- `models_storage` — артефакты моделей (`models/`)

## Train

Тренировка запускается одной командой; конфиги — Hydra.

```bash
mlflow server --host 127.0.0.1 --port 8080 \
    --backend-store-uri ./mlruns --default-artifact-root ./mlruns

python scripts/train.py

python scripts/train.py model=baseline

python scripts/train.py training.epochs=10 data.batch_size=64
```

Если MLflow-сервер недоступен, `train.py` автоматически фолбэкается на
файловый трекинг в `./mlruns`.

### Экспорт в ONNX

```bash
python scripts/export.py inference.checkpoint_path=models/best.ckpt
```

### Инференс

Точка входа — `infer.py` в корне:

```bash
python infer.py inference.image_path=path/to/image.jpg
```

Возвращает JSON с предсказанным классом и вероятностями.
Формат входа: одиночное RGB-изображение (jpg/png)

## Logging

- MLflow (`configs/config.yaml: mlflow.tracking_uri`, по умолчанию
  `http://127.0.0.1:8080`).
- Логируются: все гиперпараметры (Hydra-конфиг целиком), git commit id (тег),
  train/val/test loss и 5 метрик по эпохам, чекпойнты модели (`log_model=True`).
- В артефакты MLflow попадают: confusion matrix по эпохам, кривые обучения
  для loss и каждой метрики.
- После `trainer.fit()` графики дополнительно копируются в `plots/`.

## Overall

Структура репозитория:

```
disney-classification/
├── configs/                  # Hydra-конфиги
│   ├── config.yaml           # корневой, defaults + mlflow uri
│   ├── data/default.yaml
│   ├── model/{baseline,main}.yaml
│   ├── training/default.yaml
│   └── inference/default.yaml
├── src/disney_clf/           # python-пакет
│   ├── data/                 # dataset, datamodule, preprocess, download_data
│   ├── models/               # baseline CNN, EfficientNet
│   ├── training/module.py    # LightningModule + логирование
│   └── inference/            # ONNX-экспорт, predict
├── scripts/                  # CLI-обвязки (Hydra entrypoints)
│   ├── download_data.py
│   ├── preprocess.py
│   ├── train.py
│   └── export.py
├── infer.py                  # точка входа инференса (Hydra)
├── tests/                    # pytest
├── plots/                    # графики обучения (создаётся после train)
├── models/                   # чекпойнты + onnx (под dvc)
├── data/                     # train.csv, test.csv, raw/ (под dvc)
├── .dvc/config               # data_storage + models_storage
├── .pre-commit-config.yaml
├── pyproject.toml
└── uv.lock
```

Production-пайплайн:

1. `train.py` → `models/best.ckpt` (Lightning checkpoint) + MLflow run.
2. `export.py` → `models/model.onnx` (opset 17, dynamic batch).
3. `infer.py` → загружает ONNX через `onnxruntime`, выдаёт предсказание.
