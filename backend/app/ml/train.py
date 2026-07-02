"""Trains the OCTera diagnosis classifier.

Without --data-dir, trains on procedurally generated synthetic images as a
smoke test of the pipeline only -- the resulting checkpoint is NOT clinically
valid. Point --data-dir at a Kermany OCT2017-style layout
(``<data-dir>/train/<CLASS>/*.jpeg`` with CLASS in NORMAL/CNV/DME/DRUSEN,
e.g. https://www.kaggle.com/datasets/paultimothymooney/kermany2018) to train
on real data.

Usage:
    python -m app.ml.train --data-dir /path/to/OCT2017
    python -m app.ml.train  # synthetic smoke test
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from app.ml.features import extract_features
from app.ml.model import CLASSES, DEFAULT_CHECKPOINT_PATH, OCTClassifier
from app.ml.synthetic_dataset import generate_dataset


def _load_real_dataset(data_dir: str) -> tuple[list[np.ndarray], list[str]]:
    images: list[np.ndarray] = []
    labels: list[str] = []
    root = Path(data_dir) / "train"
    for cls in CLASSES:
        folder = root / cls
        if not folder.exists():
            continue
        for file in list(folder.glob("*.jpeg")) + list(folder.glob("*.jpg")) + list(folder.glob("*.png")):
            images.append(np.array(Image.open(file)))
            labels.append(cls)
    return images, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=str, default=None, help="Path to a Kermany OCT2017-style dataset")
    parser.add_argument("--synthetic-per-class", type=int, default=250)
    parser.add_argument("--out", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    args = parser.parse_args()

    if args.data_dir:
        raw_images, labels = _load_real_dataset(args.data_dir)
        if not raw_images:
            raise SystemExit(f"No images found under {args.data_dir}/train/<CLASS>/")
        print(f"Loaded {len(raw_images)} real images from {args.data_dir}")
    else:
        print(
            "No --data-dir given: training on synthetic procedural data. "
            "This checkpoint is a pipeline smoke test only and is NOT clinically valid."
        )
        raw_images, labels = generate_dataset(n_per_class=args.synthetic_per_class)

    X = np.stack([extract_features(Image.fromarray(img)) for img in raw_images])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = OCTClassifier()
    model.fit(X_train, y_train)

    preds = model.clf.predict(X_test)
    print(f"Validation accuracy: {accuracy_score(y_test, preds):.3f}")
    print(classification_report(y_test, preds))

    model.save(args.out)
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
