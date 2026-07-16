import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from app.ml.features import extract_features
from app.ml.model import CLASSES, OCTClassifier
from app.ml.synthetic_dataset import generate_dataset
from app.ml.train import select_best_model
from app.services.diagnosis import predict_diagnoses


def test_select_best_model_picks_a_valid_candidate_with_groups():
    """select_best_model gained a `groups` param (StratifiedGroupKFold branch) for
    patient-grouped CV, but the only prior test of this function never passed
    groups, so that branch went unexercised by any test.
    """
    raw_images, labels = generate_dataset(n_per_class=30, seed=3)
    X = np.stack([extract_features(Image.fromarray(img)) for img in raw_images])
    y = np.array(labels)
    # Each synthetic image gets its own singleton group -- a no-op grouping
    # constraint that still forces the StratifiedGroupKFold code path to run.
    groups = np.array([str(i) for i in range(len(labels))])

    best_name, best_estimator, scores = select_best_model(X, y, cv_folds=3, groups=groups)

    assert best_name in scores
    assert scores[best_name] == max(scores.values())
    assert best_estimator is not None


def test_synthetic_training_beats_random_chance():
    raw_images, labels = generate_dataset(n_per_class=60, seed=1)
    X = np.stack([extract_features(Image.fromarray(img)) for img in raw_images])
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    model = OCTClassifier()
    model.fit(X_train, y_train)
    accuracy = model.clf.score(X_test, y_test)

    assert accuracy > 1 / len(CLASSES) + 0.2


def test_select_best_model_picks_a_valid_candidate():
    raw_images, labels = generate_dataset(n_per_class=30, seed=2)
    X = np.stack([extract_features(Image.fromarray(img)) for img in raw_images])
    y = np.array(labels)

    best_name, best_estimator, scores = select_best_model(X, y, cv_folds=3)

    assert best_name in scores
    assert scores[best_name] == max(scores.values())
    assert best_estimator is not None


def test_predict_diagnoses_uses_shipped_checkpoint(tmp_path):
    img = Image.new("L", (256, 256), color=128)
    image_path = tmp_path / "sample.png"
    img.save(image_path)

    diagnoses = predict_diagnoses(str(image_path))

    assert {d.code for d in diagnoses} == set(CLASSES)
    assert abs(sum(d.probability for d in diagnoses) - 1.0) < 0.01
    assert diagnoses == sorted(diagnoses, key=lambda d: d.probability, reverse=True)
