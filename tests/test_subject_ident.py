from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from _helpers import REPO_ROOT, write_stew_recording

SCRIPT = REPO_ROOT / "Model Subject Ident" / "stewSubjectsIdentification.py"


def test_segment_recording_window_boundaries_and_shape(subject_ident_module):
    mod = subject_ident_module
    rng = np.random.default_rng(0)
    signal_values = rng.normal(size=(2000, len(mod.CHANNELS))).astype(np.float32)
    windows = mod.segment_recording(signal_values, window_seconds=4.0, step_seconds=2.0)

    window_samples = int(4.0 * mod.FS)
    step_samples = int(2.0 * mod.FS)
    expected_count = (2000 - window_samples) // step_samples + 1
    assert len(windows) == expected_count
    for window in windows:
        assert window.shape == (window_samples, len(mod.CHANNELS))


def test_segment_recording_raises_on_recording_shorter_than_one_window(subject_ident_module):
    mod = subject_ident_module
    signal_values = np.zeros((100, len(mod.CHANNELS)), dtype=np.float32)
    with pytest.raises(ValueError, match="shorter than one window"):
        mod.segment_recording(signal_values, window_seconds=4.0, step_seconds=2.0)


def test_preprocess_signal_rejects_wrong_channel_count(subject_ident_module):
    mod = subject_ident_module
    bad = np.zeros((1000, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="Expected a shape"):
        mod.preprocess_signal(bad)


def test_extract_window_features_shape_and_finite(subject_ident_module):
    mod = subject_ident_module
    rng = np.random.default_rng(1)
    window_samples = int(4.0 * mod.FS)
    window = rng.normal(size=(window_samples, len(mod.CHANNELS))).astype(np.float32)
    features = mod.extract_window_features(window)
    assert features.shape == (len(mod.FEATURE_COLUMNS),)
    assert np.all(np.isfinite(features))


def test_feature_table_preserves_window_metadata(tmp_path, subject_ident_module):
    mod = subject_ident_module
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    n_samples = int(4.0 * mod.FS) * 3  # enough for a few overlapping windows
    for subject in ("01", "02"):
        write_stew_recording(dataset_dir / f"sub{subject}_hi.txt", n_samples, seed=1)
        write_stew_recording(dataset_dir / f"sub{subject}_lo.txt", n_samples, seed=2)

    files = mod.load_recordings(dataset_dir)
    table = mod.extract_features_from_recordings(files, window_seconds=4.0, step_seconds=2.0)

    for column in ("subject_id", "condition", "recording_id", "window_index", "start_time", "label"):
        assert column in table.columns
    assert set(table["subject_id"].unique()) == {0, 1}
    assert set(table["condition"].unique()) == {"rest", "high"}
    # start_time should increase monotonically with window_index within a recording.
    one_recording = table[table["recording_id"] == "sub01_hi.txt"].sort_values("window_index")
    assert (one_recording["start_time"].diff().dropna() > 0).all()


def test_load_recordings_raises_on_empty_dataset_dir(tmp_path, subject_ident_module):
    with pytest.raises(ValueError, match="No EEG recordings found"):
        subject_ident_module.load_recordings(tmp_path)


def test_load_recordings_raises_when_subject_missing_a_condition(tmp_path, subject_ident_module):
    write_stew_recording(tmp_path / "sub01_hi.txt", 600)
    with pytest.raises(ValueError, match="missing a rest or high-workload"):
        subject_ident_module.load_recordings(tmp_path)


def test_load_feature_table_raises_on_missing_file(tmp_path, subject_ident_module):
    with pytest.raises(FileNotFoundError):
        subject_ident_module.load_feature_table(tmp_path / "missing.csv")


def _synthetic_feature_table(mod, *, n_subjects: int = 3, n_windows: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for subject in range(n_subjects):
        for window_index in range(n_windows):
            row = {
                "subject_id": subject,
                "condition": "rest" if window_index % 2 == 0 else "high",
                "recording_id": f"sub{subject}",
                "window_index": window_index,
                "start_time": float(window_index * 2.0),
                "label": subject,
            }
            for column in mod.FEATURE_COLUMNS:
                row[column] = float(rng.normal() + subject * 0.1)
            rows.append(row)
    return pd.DataFrame(rows)


def test_split_temporal_blocks_has_no_window_overlap_between_partitions(subject_ident_module):
    mod = subject_ident_module
    table = _synthetic_feature_table(mod, n_subjects=3, n_windows=9)
    train, validation, test = mod.split_temporal_blocks(table, split_count=3)

    assert len(train) + len(validation) + len(test) == len(table)
    for subject in table["subject_id"].unique():
        train_idx = set(train.loc[train["subject_id"] == subject, "window_index"])
        val_idx = set(validation.loc[validation["subject_id"] == subject, "window_index"])
        test_idx = set(test.loc[test["subject_id"] == subject, "window_index"])
        # Disjoint: no window index used in more than one partition for this subject.
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)
        # Temporal ordering: every training window index precedes every validation
        # window index, which precedes every test window index (no interleaving that
        # would let adjacent/overlapping windows cross a partition boundary).
        assert max(train_idx) < min(val_idx) < min(test_idx)


def test_split_temporal_blocks_raises_with_too_few_windows(subject_ident_module):
    mod = subject_ident_module
    table = _synthetic_feature_table(mod, n_subjects=2, n_windows=2)
    with pytest.raises(ValueError, match="fewer than"):
        mod.split_temporal_blocks(table, split_count=3)


def test_split_temporal_blocks_rejects_small_split_count(subject_ident_module):
    mod = subject_ident_module
    table = _synthetic_feature_table(mod)
    with pytest.raises(ValueError, match="split_count must be at least 3"):
        mod.split_temporal_blocks(table, split_count=2)


def test_cli_help() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "--epochs" in result.stdout
    assert "--batch-size" in result.stdout


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path, subject_ident_module):
    """Reduced synthetic run of the full pipeline: extraction -> split -> train -> plots."""
    mod = subject_ident_module
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    n_samples = int(4.0 * mod.FS) * 3
    for subject in ("01", "02", "03"):
        write_stew_recording(dataset_dir / f"sub{subject}_hi.txt", n_samples, seed=int(subject))
        write_stew_recording(dataset_dir / f"sub{subject}_lo.txt", n_samples, seed=int(subject) + 100)

    args = mod.parse_args(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--feature-csv",
            str(tmp_path / "features.csv"),
            "--output-dir",
            str(tmp_path / "out"),
            "--extract-features",
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--seed",
            "1",
            "--split-count",
            "3",
        ]
    )
    summary = mod.run_pipeline(args)

    assert summary["subject_count"] == 3
    assert summary["window_count"] > 0
    assert "test" in summary["evaluation_metrics"]
    assert (tmp_path / "out" / "metrics.json").exists()
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "confusion_matrix.png").exists()
