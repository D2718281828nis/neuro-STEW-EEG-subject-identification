#!/usr/bin/env python3
"""Classical EEG subject-identification pipeline with modular preprocessing and evaluation.

The script loads STEW EEG recordings, segments them into overlapping windows,
extracts per-window statistical, spectral, entropy, and fractal features, trains an
artificial neural network to predict subject identity from windows, and writes
evaluation metrics and plots.

Train/validation/test partitions are disjoint temporal blocks within each subject
(not leave-subject-out, since subject identity is the prediction target), so
overlapping or adjacent windows cannot cross a partition boundary.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import antropy as an
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers, models, regularizers
from keras.utils import to_categorical
from scipy.signal import butter, filtfilt, welch
from scipy.stats import kurtosis, skew
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

# Running this file directly (`python "Model Subject Ident/stewSubjectsIdentification.py"`)
# puts this file's own directory on sys.path[0], not the repo root, so the shared
# eeg_config module at the repo root would not otherwise be importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_config import CHANNELS, FS, STEP_SECONDS, WINDOW_SAMPLES, set_seed  # noqa: E402

LOGGER = logging.getLogger(__name__)

# Antropy's spectral_entropy takes an explicit sampling-frequency argument that the
# original pipeline set to 100 (not the true 128 Hz acquisition rate). Preserved here
# to keep the feature definition identical rather than silently changing its scale.
SPECTRAL_ENTROPY_SF = 100

FEATURE_COLUMNS = [
    "mean_psd",
    "std_psd",
    "amp_mean",
    "amp_std",
    "amp_var",
    "amp_range",
    "amp_skew",
    "amp_kurtosis",
    "perm_entropy",
    "spectral_entropy",
    "svd_entropy",
    "approx_entropy",
    "sample_entropy",
    "petrosian_fd",
    "katz_fd",
    "higuchi_fd",
    "detrended_fluctuation",
]


@dataclass(frozen=True)
class WindowMetadata:
    """Metadata attached to every extracted EEG window."""

    subject_id: int
    condition: str
    recording_id: str
    window_index: int
    start_time: float


def configure_logging() -> None:
    """Configure process-wide logging to stderr."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def resolve_path(path_value: str | Path, *, repo_root: Path) -> Path:
    """Resolve a CLI path relative to the repository root when it is not absolute."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def load_recordings(dataset_dir: Path) -> list[Path]:
    """Return every STEW recording file, validating that each subject has both conditions."""
    files = sorted(dataset_dir.glob("sub??_*.txt"))
    if not files:
        raise ValueError(f"No EEG recordings found in {dataset_dir}")

    subject_ids = sorted({int(path.stem[3:5]) for path in files})
    for subject_id in subject_ids:
        rest_files = [p for p in files if p.name.startswith(f"sub{subject_id:02d}_") and "_lo" in p.name]
        high_files = [p for p in files if p.name.startswith(f"sub{subject_id:02d}_") and "_hi" in p.name]
        if not rest_files or not high_files:
            raise ValueError(f"Subject {subject_id} is missing a rest or high-workload recording")

    return files


def preprocess_signal(signal_values: np.ndarray) -> np.ndarray:
    """Bandpass-filter and despike a whole recording, per channel.

    Mirrors the original pipeline: a first-order 3-40 Hz Butterworth bandpass
    (zero-phase), followed by two-stage despiking that replaces samples above the
    97th percentile of the filtered signal with its mean, then samples below the
    5th percentile of that result with its own mean.

    Args:
        signal_values: Array of shape (samples, channels).

    Returns:
        Despiked, filtered array of the same shape, dtype float32.
    """
    data = np.asarray(signal_values, dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != len(CHANNELS):
        raise ValueError(f"Expected a shape (samples, {len(CHANNELS)}) array, got {data.shape}")

    nyquist = 0.5 * FS
    normal_cutoff = [3.0 / nyquist, 40.0 / nyquist]
    b, a = butter(1, normal_cutoff, btype="bandpass", analog=False)

    despiked = np.empty_like(data)
    for channel_index in range(data.shape[1]):
        filtered = filtfilt(b, a, data[:, channel_index])
        high_threshold = np.quantile(filtered, 0.97)
        stage_one = np.where(filtered < high_threshold, filtered, filtered.mean())
        low_threshold = np.quantile(stage_one, 0.05)
        stage_two = np.where(stage_one > low_threshold, stage_one, stage_one.mean())
        despiked[:, channel_index] = stage_two

    return despiked.astype(np.float32)


def segment_recording(
    signal_values: np.ndarray,
    *,
    window_seconds: float,
    step_seconds: float,
) -> list[np.ndarray]:
    """Create complete, non-truncated overlapping windows from one recording.

    Returns a list of window arrays with shape (window_samples, channels). Any
    trailing partial window is dropped rather than silently truncated.
    """
    window_samples = int(round(window_seconds * FS))
    step_samples = int(round(step_seconds * FS))
    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("Window and step sizes must be positive")
    if signal_values.shape[0] < window_samples:
        raise ValueError(f"Recording is shorter than one window: {signal_values.shape[0]} < {window_samples}")

    starts = np.arange(0, signal_values.shape[0] - window_samples + 1, step_samples)
    return [signal_values[start : start + window_samples] for start in starts]


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """Compute the 17 channel-averaged EEG features for one window.

    Args:
        window: Array of shape (window_samples, channels).

    Returns:
        1-D array of length ``len(FEATURE_COLUMNS)`` with each feature averaged
        across channels, matching the original pipeline's channel-aggregation step.
    """
    features_per_channel: list[np.ndarray] = []
    for channel_values in window.T:
        f, power = welch(channel_values, fs=FS)
        mask = (f >= 1.0) & (f <= 45.0)
        psd = power[mask]
        features_per_channel.append(
            np.array(
                [
                    float(np.mean(psd)),
                    float(np.std(psd)),
                    float(np.mean(channel_values)),
                    float(np.std(channel_values)),
                    float(np.var(channel_values)),
                    float(np.ptp(channel_values)),
                    float(skew(channel_values)),
                    float(kurtosis(channel_values)),
                    float(an.perm_entropy(channel_values, order=3, normalize=True)),
                    float(an.spectral_entropy(channel_values, SPECTRAL_ENTROPY_SF, method="welch", normalize=True)),
                    float(an.svd_entropy(channel_values, order=3, delay=1, normalize=True)),
                    float(an.app_entropy(channel_values, order=2, metric="chebyshev")),
                    float(an.sample_entropy(channel_values, order=2, metric="chebyshev")),
                    float(an.petrosian_fd(channel_values)),
                    float(an.katz_fd(channel_values)),
                    float(an.higuchi_fd(channel_values, kmax=10)),
                    float(an.detrended_fluctuation(channel_values)),
                ],
                dtype=np.float64,
            )
        )
    return np.mean(np.stack(features_per_channel, axis=0), axis=0)


def extract_features_from_recordings(
    recordings: list[Path],
    *,
    window_seconds: float,
    step_seconds: float,
) -> pd.DataFrame:
    """Extract one feature row per window across a collection of recordings."""
    rows: list[dict[str, Any]] = []
    for recording_path in recordings:
        match = re.fullmatch(r"sub(\d{2})_(lo|hi)\.txt", recording_path.name)
        if not match:
            raise ValueError(f"Unexpected filename format: {recording_path.name}")
        subject_id = int(match.group(1)) - 1
        condition = "rest" if match.group(2) == "lo" else "high"
        data = np.loadtxt(recording_path)
        if data.ndim != 2 or data.shape[1] != len(CHANNELS):
            raise ValueError(f"Unexpected shape in {recording_path}: {data.shape}")
        processed = preprocess_signal(data)
        windows = segment_recording(processed, window_seconds=window_seconds, step_seconds=step_seconds)
        LOGGER.info("Extracting features for %s (%d windows)", recording_path.name, len(windows))
        for window_index, window in enumerate(windows):
            metadata = WindowMetadata(
                subject_id=subject_id,
                condition=condition,
                recording_id=recording_path.name,
                window_index=window_index,
                start_time=float(window_index * step_seconds),
            )
            features = extract_window_features(window)
            rows.append(
                {
                    **asdict(metadata),
                    "label": subject_id,
                    **{column: features[index] for index, column in enumerate(FEATURE_COLUMNS)},
                }
            )
    return pd.DataFrame(rows)


def load_feature_table(feature_csv: Path) -> pd.DataFrame:
    """Load a previously extracted feature table from disk."""
    if not feature_csv.exists():
        raise FileNotFoundError(f"Feature CSV not found: {feature_csv}")
    return pd.read_csv(feature_csv)


def save_feature_table(feature_table: pd.DataFrame, path: Path) -> None:
    """Persist the feature table to disk as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(path, index=False)


def split_temporal_blocks(
    feature_table: pd.DataFrame, *, split_count: int = 3
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split each subject's windows into contiguous temporal blocks for train/validation/test.

    Because subject identity is the classification target, leave-subject-out
    validation is not applicable here. Instead every subject contributes disjoint,
    time-ordered blocks of its own windows to each partition, so overlapping or
    adjacent windows never cross a partition boundary.
    """
    if split_count < 3:
        raise ValueError("split_count must be at least 3")

    train_rows: list[pd.DataFrame] = []
    validation_rows: list[pd.DataFrame] = []
    test_rows: list[pd.DataFrame] = []
    for subject_id in sorted(feature_table["subject_id"].unique()):
        subject_rows = (
            feature_table.loc[feature_table["subject_id"] == subject_id]
            .sort_values(["window_index", "start_time"])
            .copy()
        )
        if len(subject_rows) < split_count:
            raise ValueError(
                f"Subject {subject_id} has fewer than {split_count} windows; "
                "temporal-block validation requires at least split_count windows per subject"
            )
        blocks = np.array_split(subject_rows.index.to_numpy(), split_count)
        train_rows.append(subject_rows.loc[blocks[0]])
        validation_rows.append(subject_rows.loc[blocks[1]])
        test_rows.append(pd.concat([subject_rows.loc[block] for block in blocks[2:]], axis=0))

    return (
        pd.concat(train_rows, axis=0).reset_index(drop=True),
        pd.concat(validation_rows, axis=0).reset_index(drop=True),
        pd.concat(test_rows, axis=0).reset_index(drop=True),
    )


def build_classifier(n_features: int, n_classes: int) -> models.Sequential:
    """Construct the subject-identification ANN.

    Architecture mirrors the original pipeline: four ReLU hidden layers (200, 150,
    100, 75 units, L1-regularized first layer) feeding a softmax output over
    subjects, trained with Adam and categorical cross-entropy.
    """
    classifier = models.Sequential(
        [
            layers.Input(shape=(n_features,)),
            layers.Dense(200, activation="relu", kernel_regularizer=regularizers.l1(0.01)),
            layers.Dense(150, activation="relu"),
            layers.Dense(100, activation="relu"),
            layers.Dense(75, activation="relu"),
            layers.Dense(n_classes, activation="softmax"),
        ]
    )
    classifier.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return classifier


def train_and_evaluate(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    *,
    seed: int,
    epochs: int,
    batch_size: int,
) -> dict[str, Any]:
    """Fit the ANN on the training blocks and evaluate on validation and test partitions.

    Scaling is fit on the training partition only and applied to validation/test to
    avoid leaking their statistics into the model.
    """
    set_seed(seed)
    feature_columns = [column for column in FEATURE_COLUMNS if column in train_frame.columns]
    n_classes = int(pd.concat([train_frame["label"], validation_frame["label"], test_frame["label"]]).nunique())

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_frame[feature_columns].to_numpy())
    validation_features = scaler.transform(validation_frame[feature_columns].to_numpy())
    test_features = scaler.transform(test_frame[feature_columns].to_numpy())

    train_labels = train_frame["label"].to_numpy()
    validation_labels = validation_frame["label"].to_numpy()
    test_labels = test_frame["label"].to_numpy()

    classifier = build_classifier(n_features=len(feature_columns), n_classes=n_classes)
    LOGGER.info("Training classifier: %d train windows, %d epochs, batch size %d", len(train_frame), epochs, batch_size)
    history = classifier.fit(
        train_features,
        to_categorical(train_labels, num_classes=n_classes),
        validation_data=(validation_features, to_categorical(validation_labels, num_classes=n_classes)),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    validation_predictions = np.argmax(classifier.predict(validation_features, verbose=0), axis=1)
    test_predictions = np.argmax(classifier.predict(test_features, verbose=0), axis=1)

    metrics = {
        "validation": _classification_metrics(validation_labels, validation_predictions),
        "test": _classification_metrics(test_labels, test_predictions),
    }
    return {"metrics": metrics, "feature_columns": feature_columns, "history": history.history}


def _classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    """Compute window-level classification metrics with scikit-learn."""
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_macro": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
    }


def save_plots_and_metrics(output_dir: Path, evaluation: dict[str, Any]) -> None:
    """Persist training curves, confusion matrices, and a metrics summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluation["metrics"]
    history = evaluation["history"]

    if "accuracy" in history and "val_accuracy" in history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["accuracy"], label="training acc")
        ax.plot(history["val_accuracy"], label="validation acc")
        ax.set_title("Training and validation accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_validation_acc.png")
        plt.close(fig)

    if "loss" in history and "val_loss" in history:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["loss"], label="training loss")
        ax.plot(history["val_loss"], label="validation loss")
        ax.set_title("Training and validation loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_validation_loss.png")
        plt.close(fig)

    cm = np.array(metrics["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix (test)")
    ax.set_xlabel("Predicted subject")
    ax.set_ylabel("True subject")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png")
    plt.close(fig)

    metrics_summary = {
        split_name: {key: value for key, value in metrics[split_name].items() if key != "confusion_matrix"}
        for split_name in metrics
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")


def build_summary(
    *,
    args: argparse.Namespace,
    dataset_files: list[Path],
    feature_table: pd.DataFrame,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Create a machine-readable summary for the experiment."""
    dependency_versions = {
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scipy": version("scipy"),
        "scikit-learn": version("scikit-learn"),
        "matplotlib": version("matplotlib"),
        "antropy": version("antropy"),
        "keras": version("keras"),
    }
    return {
        "cli_arguments": {
            "dataset_dir": str(args.dataset_dir),
            "feature_csv": str(args.feature_csv),
            "output_dir": str(args.output_dir),
            "seed": args.seed,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "extract_features": args.extract_features,
            "window_size": args.window_size,
            "step_size": args.step_size,
            "split_count": args.split_count,
        },
        "seed": args.seed,
        "dataset_file_count": len(dataset_files),
        "subject_count": int(feature_table["subject_id"].nunique()),
        "window_count": int(len(feature_table)),
        "dependency_versions": dependency_versions,
        "evaluation_metrics": metrics,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Run the subject-identification workflow end to end."""
    repo_root = Path(__file__).resolve().parents[1]
    dataset_dir = resolve_path(args.dataset_dir, repo_root=repo_root)
    feature_csv = resolve_path(args.feature_csv, repo_root=repo_root)
    output_dir = resolve_path(args.output_dir, repo_root=repo_root)

    if args.epochs <= 0:
        raise ValueError("epochs must be > 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if args.window_size <= 0:
        raise ValueError("window_size must be > 0")
    if args.step_size <= 0:
        raise ValueError("step_size must be > 0")
    if args.split_count < 3:
        raise ValueError("split_count must be at least 3")

    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    if args.extract_features:
        dataset_files = load_recordings(dataset_dir)
        feature_table = extract_features_from_recordings(
            dataset_files, window_seconds=args.window_size, step_seconds=args.step_size
        )
        save_feature_table(feature_table, feature_csv)
    else:
        feature_table = load_feature_table(feature_csv)
        dataset_files = sorted(dataset_dir.glob("sub??_*.txt")) if dataset_dir.exists() else []

    feature_table = feature_table.copy()
    feature_table["label"] = feature_table["label"].astype(int)
    feature_table["subject_id"] = feature_table["subject_id"].astype(int)
    feature_table["window_index"] = feature_table["window_index"].astype(int)
    feature_table["start_time"] = feature_table["start_time"].astype(float)

    train_frame, validation_frame, test_frame = split_temporal_blocks(feature_table, split_count=args.split_count)
    evaluation = train_and_evaluate(
        train_frame, validation_frame, test_frame, seed=args.seed, epochs=args.epochs, batch_size=args.batch_size
    )
    save_plots_and_metrics(output_dir, evaluation)

    summary = build_summary(
        args=args, dataset_files=dataset_files, feature_table=feature_table, metrics=evaluation["metrics"]
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the subject-identification pipeline."""
    parser = argparse.ArgumentParser(description="Run the STEW EEG subject-identification pipeline")
    parser.add_argument(
        "--dataset-dir", type=Path, default=Path("dataset"), help="Directory containing STEW recordings"
    )
    parser.add_argument(
        "--feature-csv",
        type=Path,
        default=Path("Model Subject Ident/extractedFeatures.csv"),
        help="Path for the feature table (read when --extract-features is not set, written otherwise)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("Model Subject Ident/out"), help="Directory for plots and metrics"
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--epochs", type=int, default=30, help="Number of ANN training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="ANN training batch size")
    parser.add_argument(
        "--extract-features",
        action="store_true",
        help="Re-extract features from the dataset instead of reusing an existing feature CSV",
    )
    parser.add_argument("--window-size", type=float, default=WINDOW_SAMPLES / FS, help="Window size in seconds")
    parser.add_argument("--step-size", type=float, default=STEP_SECONDS, help="Step size in seconds")
    parser.add_argument(
        "--split-count",
        type=int,
        default=3,
        help="Number of contiguous temporal blocks per subject (blocks beyond the first two are pooled into test)",
    )
    return parser.parse_args(argv)


def main() -> None:
    """Entry point for the subject-identification pipeline."""
    configure_logging()
    args = parse_args()
    summary = run_pipeline(args)
    LOGGER.info("Subject-identification summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
