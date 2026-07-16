"""Unit test for the patient-grouped train/test split added in ML round 10.

Round 10 found that ~25% of OCT-AND-EYE-FUNDUS-DATASET images share a patient
with another image (both eyes and/or repeat visits), and the previous plain
stratified split let 79 of 831 patients' images leak across train and test --
inflating the reported test metrics (DME precision 0.92 -> honest 0.84).
"""

import numpy as np

from app.ml.train import _group_aware_split


def test_group_aware_split_never_splits_a_group_across_train_and_test():
    rng = np.random.default_rng(0)
    # 50 groups, most with 1 image but some with 2-4 (like the real dataset's
    # multi-visit/both-eyes patients), each group entirely NORMAL or entirely DME.
    labels: list[str] = []
    groups: list[str] = []
    for i in range(50):
        label = "DME" if i % 5 == 0 else "NORMAL"  # ~20% DME groups
        for _ in range(rng.integers(1, 5)):
            labels.append(label)
            groups.append(str(i))

    train_idx, test_idx = _group_aware_split(labels, groups, n_splits=5)

    groups_arr = np.array(groups)
    train_groups = set(groups_arr[train_idx])
    test_groups = set(groups_arr[test_idx])
    assert train_groups.isdisjoint(test_groups)
    assert len(train_idx) + len(test_idx) == len(labels)


def test_group_aware_split_keeps_both_classes_in_test():
    rng = np.random.default_rng(1)
    labels = ["DME"] * 30 + ["NORMAL"] * 170
    groups = [str(i) for i in range(200)]  # every image its own group (no multi-image patients)
    order = rng.permutation(200)
    labels = [labels[i] for i in order]
    groups = [groups[i] for i in order]

    _, test_idx = _group_aware_split(labels, groups, n_splits=5)

    test_labels = {labels[i] for i in test_idx}
    assert test_labels == {"DME", "NORMAL"}
