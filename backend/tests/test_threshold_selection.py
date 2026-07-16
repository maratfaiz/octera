"""Unit tests for the DME threshold-selection helpers added in ML rounds 5-6.

These were previously only exercised end-to-end via `python -m app.ml.train
--data-dir ...` against the real dataset -- worth having fast, data-free
coverage given a bug already slipped through here once (round 5's leaky
threshold selection, fixed in round 6).
"""

import numpy as np

from app.ml.train import _metrics_at_threshold, _oversample_minority, _select_thresholds


def test_select_thresholds_best_f1_separates_classes():
    y_true = np.array(["DME"] * 5 + ["NORMAL"] * 5)
    proba = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1, 0.05])

    result = _select_thresholds(y_true, proba, min_recall=0.6)

    # The classes are perfectly separated at 0.55/0.4 -- best F1 should land in that gap.
    assert 0.4 < result["best_f1_threshold"] <= 0.55


def test_select_thresholds_high_recall_threshold_actually_meets_target():
    y_true = np.array(["DME"] * 6 + ["NORMAL"] * 14)
    rng = np.random.default_rng(0)
    dme_proba = np.array([0.9, 0.8, 0.6, 0.5, 0.3, 0.1])
    normal_proba = rng.uniform(0, 0.4, size=14)
    proba = np.concatenate([dme_proba, normal_proba])

    result = _select_thresholds(y_true, proba, min_recall=0.8)

    assert result["high_recall_threshold"] is not None
    preds = np.where(proba >= result["high_recall_threshold"], "DME", "NORMAL")
    recall = np.mean(preds[y_true == "DME"] == "DME")
    assert recall >= 0.8


def test_select_thresholds_high_recall_is_effectively_always_reachable():
    """precision_recall_curve's lowest threshold always classifies everything as
    positive, so recall=1.0 there -- meaning high_recall_threshold is None only
    for a min_recall above 1.0. This isn't a bug to fix, just a property worth
    a test so nobody "fixes" the None branch into something that can't fire.
    """
    y_true = np.array(["DME", "DME", "NORMAL", "NORMAL", "NORMAL"])
    proba = np.array([0.9, 0.1, 0.05, 0.04, 0.01])

    result = _select_thresholds(y_true, proba, min_recall=0.99)

    assert result["high_recall_threshold"] is not None


def test_metrics_at_threshold_matches_manual_confusion_matrix():
    y_true = np.array(["DME", "DME", "NORMAL", "NORMAL"])
    proba = np.array([0.9, 0.3, 0.6, 0.1])

    result = _metrics_at_threshold(y_true, proba, threshold=0.5)

    # threshold=0.5 -> preds = [DME, NORMAL, DME, NORMAL]: 1 true DME, 1 false DME,
    # 1 false NORMAL (missed), 1 true NORMAL.
    assert result["threshold"] == 0.5
    dme_report = result["classification_report"]["DME"]
    assert dme_report["recall"] == 0.5
    assert dme_report["precision"] == 0.5


def test_oversample_minority_tops_up_without_touching_majority():
    rng = np.random.default_rng(0)
    normal_images = [np.full((8, 8), i, dtype=np.uint8) for i in range(8)]
    dme_images = [np.full((8, 8), 100 + i, dtype=np.uint8) for i in range(2)]
    images = normal_images + dme_images
    labels = ["NORMAL"] * 8 + ["DME"] * 2

    out_images, out_labels, out_groups = _oversample_minority(images, labels, rng, max_factor=3)

    assert out_labels.count("NORMAL") == 8
    assert out_labels.count("DME") == 6  # min(majority=8, minority=2 * factor=3) = 6
    assert len(out_images) == len(out_labels) == 14
    assert out_groups is None


def test_oversample_minority_caps_at_majority_size():
    rng = np.random.default_rng(0)
    normal_images = [np.full((8, 8), i, dtype=np.uint8) for i in range(5)]
    dme_images = [np.full((8, 8), 100 + i, dtype=np.uint8) for i in range(4)]
    images = normal_images + dme_images
    labels = ["NORMAL"] * 5 + ["DME"] * 4

    out_images, out_labels, _ = _oversample_minority(images, labels, rng, max_factor=10)

    # 4 * 10 = 40, capped at the majority's 5.
    assert out_labels.count("DME") == 5
    assert out_labels.count("NORMAL") == 5


def test_oversample_minority_keeps_augmented_copies_grouped_with_their_source():
    """An earlier version of this flag dropped patient-grouping entirely for the
    whole training split whenever oversampling ran, letting near-duplicate
    augmented copies of the same source image land in different CV folds --
    reintroducing the exact same-image leakage patient-grouping was built to
    fix. Augmented copies must carry their source image's group instead.
    """
    rng = np.random.default_rng(0)
    normal_images = [np.full((8, 8), i, dtype=np.uint8) for i in range(8)]
    dme_images = [np.full((8, 8), 100 + i, dtype=np.uint8) for i in range(2)]
    images = normal_images + dme_images
    labels = ["NORMAL"] * 8 + ["DME"] * 2
    groups = [f"patient-{i}" for i in range(8)] + ["patient-dme-0", "patient-dme-1"]

    out_images, out_labels, out_groups = _oversample_minority(images, labels, rng, groups=groups, max_factor=3)

    assert out_groups is not None
    assert len(out_groups) == len(out_images) == len(out_labels)
    # Every augmented DME copy's group must be one of the original DME patients' --
    # never an unrelated/unset group.
    original_dme_groups = {"patient-dme-0", "patient-dme-1"}
    for label, group in zip(out_labels[10:], out_groups[10:]):
        assert label == "DME"
        assert group in original_dme_groups
