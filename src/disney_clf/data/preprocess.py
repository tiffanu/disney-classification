from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

CLASSES = [
    "donald_duck",
    "mickey_mouse",
    "minion",
    "olaf",
    "winnie_the_pooh",
    "pumba",
]

_FOLDER_TO_CLASS = {
    "donald": "donald_duck",
    "donald_duck": "donald_duck",
    "mickey": "mickey_mouse",
    "mickey_mouse": "mickey_mouse",
    "minion": "minion",
    "olaf": "olaf",
    "pooh": "winnie_the_pooh",
    "winnie_the_pooh": "winnie_the_pooh",
    "pumba": "pumba",
}

_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}


def _collect_from_dir(class_dir: Path, class_name: str, label_idx: int) -> list:
    records = []
    for fpath in class_dir.iterdir():
        if fpath.is_file() and fpath.suffix.lower() in _EXTENSIONS:
            records.append(
                {
                    "path": str(fpath),
                    "label": label_idx,
                    "class_name": class_name,
                }
            )
    return records


def build_manifest(raw_dir: str, processed_dir: str, test_size: float = 0.2):
    class_to_label = {c: i for i, c in enumerate(CLASSES)}
    train_records, test_records = [], []

    raw = Path(raw_dir)
    cartoon_dir = raw / "cartoon"
    if cartoon_dir.is_dir():
        for split_name, target_list in [
            ("train", train_records),
            ("test", test_records),
        ]:
            split_dir = cartoon_dir / split_name
            if not split_dir.is_dir():
                continue
            for folder in split_dir.iterdir():
                if not folder.is_dir():
                    continue
                class_name = _FOLDER_TO_CLASS.get(folder.name.lower())
                if class_name is None:
                    continue
                target_list.extend(
                    _collect_from_dir(
                        folder,
                        class_name,
                        class_to_label[class_name],
                    )
                )
    else:
        all_records = []
        for folder in raw.iterdir():
            if not folder.is_dir():
                continue
            class_name = _FOLDER_TO_CLASS.get(folder.name.lower())
            if class_name is None:
                continue
            all_records.extend(
                _collect_from_dir(
                    folder,
                    class_name,
                    class_to_label[class_name],
                )
            )

        df = pd.DataFrame(all_records)
        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df["label"], random_state=42
        )
        train_records = train_df.to_dict("records")
        test_records = test_df.to_dict("records")

    train_df = pd.DataFrame(train_records)
    test_df = pd.DataFrame(test_records)

    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed / "train.csv", index=False)
    test_df.to_csv(processed / "test.csv", index=False)

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df
