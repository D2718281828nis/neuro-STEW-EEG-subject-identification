from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest
import torch
from _helpers import REPO_ROOT, write_stew_recording
from sklearn.model_selection import GroupKFold

SCRIPT = REPO_ROOT / "Model KuramotoGNN" / "kuramoto_gnn_stew.py"


# --- Graph normalization ----------------------------------------------------


def test_normalize_row_stochastic_rows_sum_to_one(kuramoto_module):
    mod = kuramoto_module
    rng = np.random.default_rng(0)
    adjacency = torch.from_numpy(rng.random((6, 6)).astype(np.float32))
    normalized = mod.normalize_row_stochastic(adjacency)
    row_sums = normalized.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones(6), atol=1e-5)


def test_normalize_row_stochastic_handles_all_zero_row(kuramoto_module):
    mod = kuramoto_module
    adjacency = torch.zeros((4, 4), dtype=torch.float32)
    adjacency[0, 1] = 1.0
    # Row 1 is entirely zero and must become a self-loop rather than divide by zero.
    normalized = mod.normalize_row_stochastic(adjacency)
    assert torch.isfinite(normalized).all()
    assert torch.allclose(normalized.sum(dim=1), torch.ones(4), atol=1e-5)
    assert normalized[1, 1] == pytest.approx(1.0)


# --- Numerical integration ---------------------------------------------------


def test_integrate_at_t_zero_returns_initial_state_without_calling_derivative(kuramoto_module):
    mod = kuramoto_module
    calls = []

    def derivative(x: torch.Tensor) -> torch.Tensor:
        calls.append(1)
        return torch.ones_like(x)

    x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    result = mod.integrate(derivative, x0, t_final=0.0, dt=0.1)
    assert torch.equal(result, x0)
    assert calls == []


def test_integrate_with_zero_derivative_is_static(kuramoto_module):
    mod = kuramoto_module

    def zero_derivative(x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    x0 = torch.tensor([[1.0, -2.0], [0.5, 3.0]], dtype=torch.float32)
    result = mod.integrate(zero_derivative, x0, t_final=5.0, dt=0.2)
    assert torch.allclose(result, x0)


# --- AUC correctness ---------------------------------------------------------


def test_compute_auc_tied_scores_both_classes_present_is_half(kuramoto_module):
    mod = kuramoto_module
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    assert mod.compute_auc(labels, scores) == pytest.approx(0.5)


def test_compute_auc_perfect_prediction_is_one(kuramoto_module):
    mod = kuramoto_module
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert mod.compute_auc(labels, scores) == pytest.approx(1.0)


def test_compute_auc_reversed_prediction_is_zero(kuramoto_module):
    mod = kuramoto_module
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    assert mod.compute_auc(labels, scores) == pytest.approx(0.0)


def test_compute_auc_single_class_is_nan(kuramoto_module):
    mod = kuramoto_module
    labels = np.array([1, 1, 1, 1])
    scores = np.array([0.1, 0.9, 0.4, 0.7])
    assert math.isnan(mod.compute_auc(labels, scores))


# --- Band power ----------------------------------------------------------


def test_relative_band_power_shape_and_range(kuramoto_module):
    mod = kuramoto_module
    rng = np.random.default_rng(0)
    windows = rng.normal(size=(5, mod.WINDOW_SAMPLES, len(mod.CHANNELS))).astype(np.float32)
    power = mod.relative_band_power(windows)
    assert power.shape == (5, len(mod.CHANNELS), len(mod.BANDS))
    assert np.isfinite(power).all()
    assert (power >= 0).all()
    assert (power <= 1.0 + 1e-6).all()


# --- Baseline normalization: metadata + zero-MAD handling -------------------


def _synthetic_record(mod, subject: int, condition: int, recording_id: str, n_windows: int, *, constant: bool = False):
    n_channels = len(mod.CHANNELS)
    n_bands = len(mod.BANDS)
    if constant:
        node_log_power = np.zeros((n_windows, n_channels, n_bands), dtype=np.float32)
    else:
        rng = np.random.default_rng(subject * 10 + condition)
        node_log_power = rng.normal(size=(n_windows, n_channels, n_bands)).astype(np.float32)
    return mod.RecordingFeatures(
        subject=subject, condition=condition, recording_id=recording_id, node_log_power=node_log_power
    )


def test_baseline_normalize_preserves_window_metadata(kuramoto_module):
    mod = kuramoto_module
    records = [
        _synthetic_record(mod, subject=1, condition=0, recording_id="sub01_lo.txt", n_windows=4),
        _synthetic_record(mod, subject=1, condition=1, recording_id="sub01_hi.txt", n_windows=4),
        _synthetic_record(mod, subject=2, condition=0, recording_id="sub02_lo.txt", n_windows=3),
        _synthetic_record(mod, subject=2, condition=1, recording_id="sub02_hi.txt", n_windows=3),
    ]
    windows, conditions, subjects, metadata, region_labels = mod.baseline_normalize(records)

    assert windows.shape[0] == len(conditions) == len(subjects) == len(metadata)
    for column in ("subject", "condition", "recording_id", "window_index", "start_time"):
        assert column in metadata.columns
    assert set(metadata["subject"]) == {1, 2}
    assert region_labels.shape == (len(mod.CHANNELS),)


def test_baseline_normalize_finite_when_rest_mad_is_zero(kuramoto_module):
    """Constant rest windows drive MAD to zero; the scale floor must keep output finite."""
    mod = kuramoto_module
    records = [
        _synthetic_record(mod, subject=1, condition=0, recording_id="sub01_lo.txt", n_windows=4, constant=True),
        _synthetic_record(mod, subject=1, condition=1, recording_id="sub01_hi.txt", n_windows=4),
    ]
    windows, conditions, subjects, metadata, region_labels = mod.baseline_normalize(records)
    assert np.isfinite(windows).all()


# --- Leakage prevention: subject-grouped CV ----------------------------------


def test_group_kfold_never_splits_a_subject_across_train_and_test():
    rng = np.random.default_rng(0)
    subjects = np.repeat(np.arange(8), 10)
    conditions = rng.integers(0, 2, size=subjects.shape)
    splitter = GroupKFold(n_splits=4)
    for train_idx, test_idx in splitter.split(np.zeros_like(subjects), conditions, groups=subjects):
        train_subjects = set(subjects[train_idx])
        test_subjects = set(subjects[test_idx])
        assert train_subjects.isdisjoint(test_subjects)


def test_train_logistic_classifier_cv_reports_fold_and_aggregate_metrics(kuramoto_module):
    mod = kuramoto_module
    rng = np.random.default_rng(0)
    n_subjects = 6
    windows_per_subject = 10
    subjects = np.repeat(np.arange(n_subjects), windows_per_subject)
    conditions = np.tile([0, 1] * (windows_per_subject // 2), n_subjects)
    features = torch.from_numpy(rng.normal(size=(len(subjects), 4)).astype(np.float32))
    fold_indices = list(GroupKFold(n_splits=3).split(np.zeros(len(subjects)), conditions, groups=subjects))

    result = mod.train_logistic_classifier_cv(features, conditions, fold_indices, seed=0)

    assert len(result["fold_metrics"]) == 3
    for fold in result["fold_metrics"]:
        assert 0.0 <= fold["accuracy"] <= 1.0
    assert 0.0 <= result["accuracy_mean"] <= 1.0
    assert result["confusion_matrix"].shape == (2, 2)


# --- CLI and smoke test -------------------------------------------------------


def test_cli_help() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0
    assert "--cv_splits" in result.stdout


@pytest.mark.slow
def test_end_to_end_smoke(tmp_path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    n_samples = 128 * 4 * 3  # a few windows at the default FS/window size
    for subject in ("01", "02", "03"):
        write_stew_recording(dataset_dir / f"sub{subject}_hi.txt", n_samples, seed=int(subject))
        write_stew_recording(dataset_dir / f"sub{subject}_lo.txt", n_samples, seed=int(subject) + 100)

    output_dir = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset",
            str(dataset_dir),
            "--output_dir",
            str(output_dir),
            "--sample_windows",
            "20",
            "--max_T",
            "2",
            "--cv_splits",
            "2",
            "--seed",
            "3",
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "window_metadata.csv").exists()
    assert (output_dir / "mean_pairwise_distance.png").exists()
