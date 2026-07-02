import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from app.ml.features import extract_features
from app.ml.model import CLASSES, OCTClassifier
from app.ml.synthetic_dataset import generate_dataset
from app.services.diagnosis import predict_diagnoses


def test_synthetic_training_beats_random_chance():
    raw_images, labels = generate_dataset(n_per_class=60, seed=1)
    X = np.stack([extract_features(Image.fromarray(img)) for img in raw_images])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    model = OCTClassifier()
    model.fit(X_train, y_train)
    accuracy = model.clf.score(X_test, y_test)

    assert accuracy > 1 / len(CLASSES) + 0.2


def test_predict_diagnoses_uses_shipped_checkpoint(tmp_path):
    img = Image.new("L", (256, 256), color=128)
    image_path = tmp_path / "sample.png"
    img.save(image_path)

    diagnoses = predict_diagnoses(str(image_path))

    assert {d.code for d in diagnoses} == set(CLASSES)
    assert abs(sum(d.probability for d in diagnoses) - 1.0) < 0.01
    assert diagnoses == sorted(diagnoses, key=lambda d: d.probability, reverse=True)
