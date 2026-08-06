from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest
from _helpers import REPO_ROOT, write_stew_recording

SCRIPT = REPO_ROOT / "Model GAT" / "stew_asi_gat_experiment.py"


def test_relative_band_power_shape_and_range(gat_module):
    mod = gat_module
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(4, mod.WINDOW_SAMPLES, len(mod.CHANNELS))).astype(np.float64)
    power = mod.relative_band_power(windows)
    assert power.shape == (4, len(mod.CHANNELS), len(mod.BANDS))
    assert np.isfinite(power).all()
    assert (power >= 0).all()


def test_validate_dataset_files_raises_when_subject_missing_a_condition(tmp_path, gat_module):
    write_stew_recording(tmp_path / "sub01_hi.txt", 19200)
    files = sorted(tmp_path.glob("sub??_*.txt"))
    with pytest.raises(ValueError, match="missing a rest or high-workload"):
        gat_module.validate_dataset_files(files)


def test_validate_dataset_files_raises_on_empty_list(gat_module):
    with pytest.raises(ValueError, match="No STEW recordings found"):
        gat_module.validate_dataset_files([])


def test_cli_help() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "--splits" in result.stdout


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    for subject in ("01", "02", "03"):
        write_stew_recording(dataset_dir / f"sub{subject}_hi.txt", 19200, seed=int(subject))
        write_stew_recording(dataset_dir / f"sub{subject}_lo.txt", 19200, seed=int(subject) + 100)

    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            str(dataset_dir),
            "--output",
            str(output_dir),
            "--seed",
            "3",
            "--splits",
            "5",
            "--bootstrap",
            "20",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "subject_condition_metrics.csv").exists()
