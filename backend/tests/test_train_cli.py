"""Covers main()'s fail-fast validation for the internal subprocess-entry-point
flags (round 30 cleanup): each pairs a cheap flag with an expensive phase, so a
future wiring bug in _multi_seed_evaluation/_run_real_training should error
immediately via parser.error() rather than surfacing confusingly deep inside a
load/augment/fit run that can take minutes to hours. Invoked via subprocess
(matching how these flags are actually used in production) rather than calling
main() in-process, since parser.error() calls sys.exit().
"""
import subprocess
import sys


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
