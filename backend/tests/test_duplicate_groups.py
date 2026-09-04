"""Unit test for ML round 14: merging patient groups that share a duplicate image.

Round 10 grouped by nominal patient ID; round 14 found 12 pairs of byte-identical
images filed under two *different* patient IDs, 2 of which still ended up split
across train and test despite the round 10 fix.
"""

from app.ml.train import _merge_duplicate_groups


def test_patients_sharing_a_duplicate_image_are_merged_into_one_group():
    patient_ids = ["1330", "1348", "1330", "1348", "2000"]
    content_hashes = ["hashA", "hashA", "hashB", "hashB", "hashC"]

    groups = _merge_duplicate_groups(patient_ids, content_hashes)

    # 1330 and 1348 share two duplicate images (hashA, hashB) -- must end up in
    # the same group despite being different nominal patients.
    assert groups[0] == groups[1] == groups[2] == groups[3]
    # 2000 shares no duplicate with anyone and stays its own group.
    assert groups[4] != groups[0]


def test_unrelated_patients_stay_in_separate_groups():
    patient_ids = ["A", "B", "C"]
    content_hashes = ["h1", "h2", "h3"]  # no duplicates at all

    groups = _merge_duplicate_groups(patient_ids, content_hashes)

    assert len(set(groups)) == 3
