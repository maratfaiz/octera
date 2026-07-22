"""Covers main()'s fail-fast validation for the internal subprocess-entry-point
flags (round 30 cleanup): each pairs a cheap flag with an expensive phase, so a
future wiring bug in _multi_seed_evaluation/_run_real_training should error
immediately via parser.error() rather than surfacing confusingly deep inside a
load/augment/fit run that can take minutes to hours. Invoked via subprocess
(matching how these flags are actually used in production) rather than calling
main() in-process, since parser.error() calls sys.exit().

Also covers _phase_subprocess_cmd's --no-*-augment flag forwarding directly
(round 40 code review flagged this as untested: a future edit that forgets to
forward a new augmentation flag here would silently make the final-refit/
select-phase subprocess always augment regardless of what the user asked for,
and nothing in the CLI-validation tests above would catch it).
"""
import argparse
import subprocess
import sys

from app.ml.train import _phase_subprocess_cmd


def _base_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        data_dir="/tmp/oct_dataset",
        out="/tmp/checkpoint.joblib",
        no_rotation_augment=False,
        no_flip_augment=False,
        no_brightness_augment=False,
        no_shift_augment=False,
        no_perspective_augment=False,
        minority_oversample=1,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_phase_subprocess_cmd_forwards_no_shift_augment_when_set():
    cmd = _phase_subprocess_cmd(_base_args(no_shift_augment=True))
    assert "--no-shift-augment" in cmd


def test_phase_subprocess_cmd_omits_no_shift_augment_by_default():
    cmd = _phase_subprocess_cmd(_base_args())
    assert "--no-shift-augment" not in cmd


def test_phase_subprocess_cmd_forwards_no_perspective_augment_when_set():
    cmd = _phase_subprocess_cmd(_base_args(no_perspective_augment=True))
    assert "--no-perspective-augment" in cmd


def test_phase_subprocess_cmd_omits_no_perspective_augment_by_default():
    cmd = _phase_subprocess_cmd(_base_args())
    assert "--no-perspective-augment" not in cmd


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "app.ml.train", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_multi_seed_worker_seed_requires_estimator():
    result = _run("--multi-seed-worker-seed", "42")
    assert result.returncode != 0
    assert "--multi-seed-worker-estimator" in result.stderr


def test_phase_select_requires_phase_output():
    result = _run("--phase", "select")
    assert result.returncode != 0
    assert "--phase-output" in result.stderr


def test_phase_final_refit_requires_estimator_name():
    result = _run("--phase", "final-refit")
    assert result.returncode != 0
    assert "--estimator-name" in result.stderr


def test_multi_seed_worker_seed_requires_data_dir():
    # Originally this fell through to _load_real_dataset's Path(None), a raw
    # TypeError traceback rather than an actionable message -- found by this
    # test itself before the --data-dir check below existed.
    result = _run("--multi-seed-worker-seed", "42", "--multi-seed-worker-estimator", "svm_rbf_pca75")
    assert result.returncode != 0
    assert "--data-dir" in result.stderr
    assert "TypeError" not in result.stderr
