"""STEW EEG graph-dynamics experiment comparing GRAND diffusion and Kuramoto coupling.

Loads STEW EEG recordings, encodes band-power windows as graph node features,
integrates two dynamics mechanisms (linear GRAND diffusion and Kuramoto phase
coupling) over a normalized channel adjacency graph, and evaluates the resulting
pooled representations with a subject-grouped cross-validated logistic classifier
so that windows from the same subject never appear in both the train and test side
of a fold.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.model_selection import GroupKFold

# Running this file directly puts its own directory on sys.path[0], not the repo
# root, so the shared eeg_config module would not otherwise be importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg_config import BANDS, CHANNELS, FS, REGIONS, STEP_SAMPLES, STEP_SECONDS, WINDOW_SAMPLES, set_seed  # noqa: E402

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindowMetadata:
    """Metadata attached to every extracted EEG window, used as CV grouping keys."""

    subject: int
    condition: int
    recording_id: str
    window_index: int
    start_time: float


@dataclass
class RecordingFeatures:
    """Per-recording node log-power features plus the metadata for each window."""

    subject: int
    condition: int
    recording_id: str
    node_log_power: np.ndarray


def configure_logging() -> None:
    """Configure process-wide logging to stderr."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def _channel_indices(names: list[str]) -> list[int]:
    return [CHANNELS.index(name) for name in names]


def _window_view(data: np.ndarray) -> np.ndarray:
    starts = np.arange(0, data.shape[0] - WINDOW_SAMPLES + 1, STEP_SAMPLES)
    return np.stack([data[start : start + WINDOW_SAMPLES] for start in starts], axis=0)


def preprocess_eeg(data: np.ndarray) -> np.ndarray:
    """Mean-center a recording per channel."""
    data = data.astype(np.float32)
    return data - data.mean(axis=0, keepdims=True)


def relative_band_power(windows: np.ndarray) -> np.ndarray:
    """Compute per-band relative power for a stack of windows.

    Args:
        windows: Array of shape (n_windows, n_samples, n_channels).

    Returns:
        Array of shape (n_windows, n_channels, n_bands) with power in each band
        normalized by the total 1-45 Hz power.
    """
    n_windows, n_samples, n_channels = windows.shape
    window_fn = np.hanning(n_samples).astype(np.float32)[None, :, None]
    windows = windows * window_fn
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / FS)
    spectrum = np.fft.rfft(windows, axis=1)
    psd = (np.abs(spectrum) ** 2) / np.sum(window_fn**2)
    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total = np.sum(psd[:, total_mask, :], axis=1)
    bands = []
    for low, high in BANDS.values():
        mask = (freqs >= low) & (freqs < high)
        bands.append(np.sum(psd[:, mask, :], axis=1))
    absolute = np.stack(bands, axis=-1)
    return absolute / np.maximum(total[:, :, None], 1e-12)


def build_node_log_power(relative_power: np.ndarray) -> np.ndarray:
    """Log-transform relative band power, floored to avoid log(0)."""
    return np.log(np.maximum(relative_power, 1e-12))


def validate_dataset_files(files: list[Path]) -> None:
    """Raise an actionable error if any subject is missing a rest or high-workload file."""
    if not files:
        raise ValueError("No STEW recordings found")
    subject_ids = sorted({int(path.stem[3:5]) for path in files})
    for subject_id in subject_ids:
        has_rest = any(path.name.startswith(f"sub{subject_id:02d}_") and "_lo" in path.name for path in files)
        has_high = any(path.name.startswith(f"sub{subject_id:02d}_") and "_hi" in path.name for path in files)
        if not has_rest or not has_high:
            raise ValueError(f"Subject {subject_id} is missing a rest or high-workload recording")


def extract_recording(path: Path) -> RecordingFeatures:
    """Load one STEW recording and compute its per-window node log-power features."""
    match = re.fullmatch(r"sub(\d{2})_(lo|hi)\.txt", path.name)
    if not match:
        raise ValueError(f"Unexpected STEW filename: {path.name}")
    subject = int(match.group(1))
    condition = 0 if match.group(2) == "lo" else 1
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != len(CHANNELS):
        raise ValueError(f"Unexpected STEW file shape for {path}: {data.shape}")
    data = preprocess_eeg(data)
    windows = _window_view(data)
    relative_power = relative_band_power(windows)
    node_log_power = build_node_log_power(relative_power)
    return RecordingFeatures(
        subject=subject, condition=condition, recording_id=path.name, node_log_power=node_log_power
    )


def get_channel_region_labels() -> np.ndarray:
    """Map each channel to an integer region id, in CHANNELS order."""
    labels = np.zeros(len(CHANNELS), dtype=np.int64)
    region_names = list(REGIONS.keys())
    for region_index, region in enumerate(region_names):
        for channel in REGIONS[region]:
            labels[CHANNELS.index(channel)] = region_index
    return labels


def baseline_normalize(
    records: list[RecordingFeatures],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """Robust-normalize node log-power against each subject's rest baseline.

    Returns:
        windows: (N, n_channels, n_bands) normalized node features.
        conditions: (N,) binary condition labels (0=rest, 1=high).
        subjects: (N,) subject id per window, used as CV grouping keys.
        metadata: DataFrame with one row per window (subject, condition,
            recording_id, window_index, start_time).
        region_labels: (n_channels,) integer region id per channel.
    """
    by_subject: dict[int, list[RecordingFeatures]] = {}
    for record in records:
        by_subject.setdefault(record.subject, []).append(record)

    all_windows: list[np.ndarray] = []
    all_conditions: list[int] = []
    all_subjects: list[int] = []
    metadata_rows: list[dict[str, Any]] = []
    for subject, recs in sorted(by_subject.items()):
        rest_windows = np.concatenate([rec.node_log_power for rec in recs if rec.condition == 0], axis=0)
        if rest_windows.shape[0] == 0:
            raise ValueError(f"Missing rest windows for subject {subject}")
        center = np.median(rest_windows, axis=0)
        mad = np.median(np.abs(rest_windows - center), axis=0)
        scale = np.maximum(mad * 1.4826, 1e-6)
        for rec in recs:
            z = (rec.node_log_power - center) / scale
            z[:, :, 1] *= -1.0
            z = np.clip(z, -6.0, 6.0)
            all_windows.append(z.astype(np.float32))
            all_conditions.extend([rec.condition] * z.shape[0])
            all_subjects.extend([subject] * z.shape[0])
            for window_index in range(z.shape[0]):
                metadata_rows.append(
                    asdict(
                        WindowMetadata(
                            subject=subject,
                            condition=rec.condition,
                            recording_id=rec.recording_id,
                            window_index=window_index,
                            start_time=float(window_index * STEP_SECONDS),
                        )
                    )
                )

    windows = np.concatenate(all_windows, axis=0)
    conditions = np.array(all_conditions, dtype=np.int64)
    subjects = np.array(all_subjects, dtype=np.int64)
    metadata = pd.DataFrame(metadata_rows)
    return windows, conditions, subjects, metadata, get_channel_region_labels()


def build_channel_graph() -> torch.Tensor:
    """Build the row-normalized channel adjacency: dense within region, sparse across."""
    n = len(CHANNELS)
    adj = np.zeros((n, n), dtype=np.float32)
    region_map = {channel: region for region, members in REGIONS.items() for channel in members}
    for i, ci in enumerate(CHANNELS):
        for j, cj in enumerate(CHANNELS):
            if i == j:
                continue
            adj[i, j] = 1.0 if region_map[ci] == region_map[cj] else 0.12
    return normalize_row_stochastic(torch.from_numpy(adj))


def normalize_row_stochastic(adj: torch.Tensor) -> torch.Tensor:
    """Row-normalize an adjacency matrix so every row sums to 1."""
    assert adj.dim() == 2 and adj.shape[0] == adj.shape[1]
    row_sum = adj.sum(dim=1, keepdim=True)
    zero_rows = row_sum.squeeze(1) == 0.0
    safe_adj = adj.clone()
    if zero_rows.any():
        safe_adj[zero_rows, :] = 0.0
        safe_adj[zero_rows, torch.arange(adj.shape[0])[zero_rows]] = 1.0
        row_sum = safe_adj.sum(dim=1, keepdim=True)
    a_hat = safe_adj / row_sum
    if not torch.allclose(a_hat.sum(dim=1), torch.ones(adj.shape[0], dtype=torch.float32), atol=1e-5):
        raise ValueError("A_hat rows do not sum to 1.")
    return a_hat


def random_encoder_weights(feature_dim: int, hidden_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a fixed random linear encoder (weights, bias) for reproducible window encoding."""
    rng = np.random.default_rng(seed)
    w = torch.from_numpy(rng.normal(scale=0.5, size=(feature_dim, hidden_dim)).astype(np.float32))
    b = torch.from_numpy(rng.normal(scale=0.1, size=(hidden_dim,)).astype(np.float32))
    return w, b


def encode_windows(windows: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Apply the random linear encoder to flattened per-window node features."""
    return windows @ w + b


def integrate(
    derivative_fn: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, t_final: float, dt: float
) -> torch.Tensor:
    """Explicit-Euler integrate node states under `derivative_fn` from 0 to `t_final`."""
    assert x0.dtype == torch.float32
    assert t_final >= 0.0 and dt > 0.0
    if t_final == 0.0:
        return x0.clone()
    x = x0.clone()
    t = 0.0
    while t < t_final:
        step = min(dt, t_final - t)
        dxdt = derivative_fn(x)
        x = x + step * dxdt
        if torch.isnan(x).any():
            raise ValueError(f"NaN detected during integration at t={t:.4f}.")
        t += step
    return x


def grand_derivative(x: torch.Tensor, a_hat: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    """Linear GRAND diffusion derivative: (A_hat - I) X."""
    return (a_hat - identity) @ x


def kuramoto_derivative(x: torch.Tensor, a_hat: torch.Tensor, omega: torch.Tensor, k: float) -> torch.Tensor:
    """Kuramoto phase-coupling derivative: omega + K * A_hat sin(X_j - X_i)."""
    diff = x.unsqueeze(0) - x.unsqueeze(1)
    coupling = torch.einsum("ij,ijd->id", a_hat, torch.sin(diff))
    return omega + k * coupling


def pairwise_distances(x: torch.Tensor) -> torch.Tensor:
    squared_norms = (x * x).sum(dim=1, keepdim=True)
    dist_sq = squared_norms + squared_norms.t() - 2.0 * (x @ x.t())
    return torch.sqrt(torch.clamp(dist_sq, min=0.0))


def mean_pairwise_distance(x: torch.Tensor) -> float:
    dist = pairwise_distances(x)
    n = x.shape[0]
    return float(dist.sum() / (n * (n - 1)))


def dirichlet_energy(x: torch.Tensor, a_hat: torch.Tensor) -> float:
    dist_sq = pairwise_distances(x) ** 2
    energy = 0.5 * (a_hat * dist_sq).sum() / x.shape[0]
    return float(energy)


def class_separation_score(x: torch.Tensor, labels: np.ndarray) -> float:
    dist = pairwise_distances(x)
    same = labels[:, None] == labels[None, :]
    diff = ~same
    same_f = same.astype(np.float32)
    diff_f = diff.astype(np.float32)
    same_sum = float((dist * torch.from_numpy(same_f)).sum()) - float(np.trace(dist.numpy()))
    diff_sum = float((dist * torch.from_numpy(diff_f)).sum())
    same_count = same_f.sum() - x.shape[0]
    diff_count = diff_f.sum()
    if same_count <= 0 or diff_count <= 0:
        return 0.0
    return float(diff_sum / diff_count - same_sum / same_count)


def average_cosine_similarity(x: torch.Tensor) -> float:
    normalized = torch.nn.functional.normalize(x, dim=1)
    sim = normalized @ normalized.t()
    n = x.shape[0]
    return float((sim.sum() - n) / (n * (n - 1)))


def velocity_synchronization(x: torch.Tensor, derivative_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:
    v = derivative_fn(x)
    dist = pairwise_distances(v)
    n = v.shape[0]
    return float(dist.sum() / (n * (n - 1)))


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC via scikit-learn (correct average-rank tie handling).

    Returns NaN when only one class is present, since AUC is undefined in that case.
    """
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores)))


def compute_metrics(
    state: torch.Tensor,
    a_hat: torch.Tensor,
    labels: np.ndarray,
    derivative_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, float]:
    """Compute graph-dynamics diagnostics (distance, energy, separation, sync) for one state."""
    metrics: dict[str, float] = {
        "mean_pairwise_distance": mean_pairwise_distance(state),
        "dirichlet_energy": dirichlet_energy(state, a_hat),
        "class_separation": class_separation_score(state, labels),
        "cosine_similarity": average_cosine_similarity(state),
        "velocity_sync": float("nan"),
    }
    if derivative_fn is not None:
        metrics["velocity_sync"] = velocity_synchronization(state, derivative_fn)
    return metrics


def sample_windows(
    windows: np.ndarray,
    conditions: np.ndarray,
    subjects: np.ndarray,
    metadata: pd.DataFrame,
    max_windows: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Subsample windows to at most `max_windows`, balanced across rest/high, keeping arrays aligned."""
    n = windows.shape[0]
    if n <= max_windows:
        return windows, conditions, subjects, metadata.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rest_idx = indices[conditions == 0]
    high_idx = indices[conditions == 1]
    count = max_windows // 2
    chosen = np.concatenate(
        [
            rng.choice(rest_idx, size=min(count, len(rest_idx)), replace=False),
            rng.choice(high_idx, size=min(count, len(high_idx)), replace=False),
        ]
    )
    rng.shuffle(chosen)
    return windows[chosen], conditions[chosen], subjects[chosen], metadata.iloc[chosen].reset_index(drop=True)


def _fit_logistic(x_train: torch.Tensor, y_train: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    dim = x_train.shape[1]
    w = torch.zeros(dim, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([w, b], lr=0.1)
    for _ in range(200):
        logits = x_train @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return w.detach(), b.detach()


def train_logistic_classifier_cv(
    features: torch.Tensor,
    conditions: np.ndarray,
    fold_indices: list[tuple[np.ndarray, np.ndarray]],
    seed: int,
) -> dict[str, Any]:
    """Subject-grouped cross-validated logistic classifier over pooled window features.

    Every subject's windows fall entirely within one fold's train or test side (never
    both), so held-out performance is not inflated by leakage across overlapping or
    adjacent windows from the same subject.
    """
    n = features.shape[0]
    labels_t = torch.from_numpy(conditions).float()
    oof_probs = np.full(n, np.nan)
    oof_preds = np.full(n, -1, dtype=int)
    fold_metrics: list[dict[str, Any]] = []

    for fold_index, (train_idx, test_idx) in enumerate(fold_indices):
        w, b = _fit_logistic(features[train_idx], labels_t[train_idx], seed=seed + fold_index)
        with torch.no_grad():
            probs = torch.sigmoid(features[test_idx] @ w + b).numpy()
        preds = (probs >= 0.5).astype(int)
        y_test = conditions[test_idx]
        oof_probs[test_idx] = probs
        oof_preds[test_idx] = preds
        fold_auc = compute_auc(y_test, probs)
        if np.isnan(fold_auc):
            LOGGER.warning("CV fold %d has a single class in its held-out windows; AUC is undefined (NaN).", fold_index)
        fold_metrics.append(
            {
                "fold": fold_index,
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y_test, preds)),
                "auc": fold_auc,
            }
        )

    accuracies = np.array([m["accuracy"] for m in fold_metrics])
    aucs = np.array([m["auc"] for m in fold_metrics if not np.isnan(m["auc"])])
    return {
        "fold_metrics": fold_metrics,
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std()),
        "auc_mean": float(aucs.mean()) if len(aucs) else float("nan"),
        "auc_std": float(aucs.std()) if len(aucs) else float("nan"),
        "overall_accuracy": float(accuracy_score(conditions, oof_preds)),
        "overall_auc": compute_auc(conditions, oof_probs),
        "confusion_matrix": sk_confusion_matrix(conditions, oof_preds, labels=[0, 1]),
    }


def plot_metric_curve(
    metric_history: dict[str, list[float]], t_values: list[float], ylabel: str, filename: Path, title: str
) -> None:
    plt.figure(figsize=(8, 5))
    for model_name, values in metric_history.items():
        plt.plot(t_values, values, marker="o", label=model_name)
    plt.xlabel("Terminal time T")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_pca_representations(
    representations: dict[str, dict[float, torch.Tensor]],
    labels: np.ndarray,
    small_t: float,
    large_t: float,
    filename: Path,
) -> None:
    model_names = list(representations.keys())
    fig, axes = plt.subplots(2, len(model_names), figsize=(5 * len(model_names), 10))
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    x0 = representations[model_names[0]][0.0].cpu().numpy()
    mean = x0.mean(axis=0, keepdims=True)
    cov = np.cov(x0 - mean, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order[:2]]

    for col, model_name in enumerate(model_names):
        for row, t_value in enumerate((small_t, large_t)):
            x = representations[model_name][t_value].cpu().numpy()
            x2 = (x - mean) @ components
            ax = axes[row, col]
            ax.scatter(x2[:, 0], x2[:, 1], c=labels, cmap="tab10", s=20, alpha=0.8)
            ax.set_title(f"{model_name}\nT={t_value}")
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.grid(True, alpha=0.2)
    fig.suptitle("PCA of pooled window representations by rest/high", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(filename)
    plt.close(fig)


def plot_channel_graph(adjacency: torch.Tensor, filename: Path) -> None:
    n = adjacency.shape[0]
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    fig, ax = plt.subplots(figsize=(7, 7))
    edges = torch.nonzero(adjacency > 0.0, as_tuple=False).cpu().numpy()
    for i, j in edges:
        if i < j:
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color="gray",
                alpha=0.3,
                linewidth=0.8,
            )
    region_labels = get_channel_region_labels()
    ax.scatter(positions[:, 0], positions[:, 1], c=region_labels, cmap="tab10", s=120, edgecolor="black")
    for i, channel in enumerate(CHANNELS):
        ax.text(positions[i, 0], positions[i, 1], channel, fontsize=8, ha="center", va="center")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("STEW EEG channel graph for KuramotoGNN")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def plot_connectivity_heatmap(adjacency: torch.Tensor, filename: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(adjacency.cpu().numpy(), cmap="viridis", aspect="equal")
    ax.set_title("Node connectivity heatmap")
    ax.set_xlabel("Target node")
    ax.set_ylabel("Source node")
    ax.set_xticks(np.arange(len(CHANNELS)))
    ax.set_yticks(np.arange(len(CHANNELS)))
    ax.set_xticklabels(CHANNELS, rotation=90, fontsize=8)
    ax.set_yticklabels(CHANNELS, fontsize=8)
    fig.colorbar(im, ax=ax, label="Normalized edge weight")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, model_name: str, t_value: float, filename: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title(f"Confusion matrix (OOF): {model_name} at T={t_value}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Rest", "High"])
    ax.set_yticklabels(["Rest", "High"])
    fig.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def print_results_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'Model':<32} {'T':>6} {'MeanDist':>10} {'Dirichlet':>12} {'ClassSep':>10} "
        f"{'CosSim':>9} {'AccMean':>9} {'AccStd':>8} {'AUCMean':>9} {'AUCStd':>8}"
    )
    LOGGER.info(header)
    LOGGER.info("-" * len(header))
    for row in results:
        LOGGER.info(
            "%-32s %6.1f %10.4f %12.4f %10.4f %9.4f %9.3f %8.3f %9.3f %8.3f",
            row["model"],
            row["T"],
            row["mean_pairwise_distance"],
            row["dirichlet_energy"],
            row["class_separation"],
            row["cosine_similarity"],
            row["accuracy_mean"],
            row["accuracy_std"],
            row["auc_mean"],
            row["auc_std"],
        )


def build_summary(
    *,
    args: argparse.Namespace,
    dataset_files: list[Path],
    subjects: np.ndarray,
    window_count: int,
    cv_splits_used: int,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a machine-readable summary for the experiment."""
    dependency_versions = {
        "numpy": version("numpy"),
        "pandas": version("pandas"),
        "scipy": version("scipy") if _has_dist("scipy") else "not installed",
        "scikit-learn": version("scikit-learn"),
        "matplotlib": version("matplotlib"),
        "torch": version("torch"),
    }
    metrics_table = [{key: value for key, value in row.items() if key != "confusion_matrix"} for row in results]
    return {
        "cli_arguments": vars(args) | {"dataset": str(args.dataset), "output_dir": str(args.output_dir)},
        "seed": args.seed,
        "dataset_file_count": len(dataset_files),
        "subject_count": int(len(np.unique(subjects))),
        "window_count": window_count,
        "cv_splits_used": cv_splits_used,
        "dependency_versions": dependency_versions,
        "evaluation_metrics": {"metric_level": "window", "results": metrics_table},
    }


def _has_dist(name: str) -> bool:
    try:
        version(name)
        return True
    except Exception:
        return False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for the KuramotoGNN STEW experiment."""
    parser = argparse.ArgumentParser(description="KuramotoGNN dynamics experiment on the STEW EEG dataset")
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--sample_windows", type=int, default=300)
    parser.add_argument("--feature_dim", type=int, default=20)
    parser.add_argument("--K", type=float, default=2.0)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--max_T", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--cv_splits", type=int, default=5, help="Number of subject-grouped cross-validation folds")
    parser.add_argument("--output_dir", type=Path, default=Path("Model KuramotoGNN"))
    return parser.parse_args(argv)


def main() -> None:
    """Run the STEW KuramotoGNN experiment end to end."""
    configure_logging()
    args = parse_args()

    if args.sample_windows <= 0:
        raise ValueError("sample_windows must be > 0")
    if args.feature_dim <= 0:
        raise ValueError("feature_dim must be > 0")
    if args.dt <= 0:
        raise ValueError("dt must be > 0")
    if args.max_T < 0:
        raise ValueError("max_T must be >= 0")
    if args.cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir = args.output_dir / "test_output"
    test_output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.dataset.glob("sub??_*.txt"))
    validate_dataset_files(files)
    records = [extract_recording(path) for path in files]
    windows, conditions, subjects, metadata, region_labels = baseline_normalize(records)
    windows, conditions, subjects, metadata = sample_windows(
        windows, conditions, subjects, metadata, args.sample_windows, args.seed
    )
    metadata.to_csv(args.output_dir / "window_metadata.csv", index=False)

    unique_subjects = np.unique(subjects)
    cv_splits_used = min(args.cv_splits, len(unique_subjects))
    if cv_splits_used < args.cv_splits:
        LOGGER.warning(
            "Requested cv_splits=%d exceeds the %d subjects present after sampling; using %d folds instead.",
            args.cv_splits,
            len(unique_subjects),
            cv_splits_used,
        )
    fold_indices = list(
        GroupKFold(n_splits=cv_splits_used).split(np.zeros(len(conditions)), conditions, groups=subjects)
    )

    windows_tensor = torch.from_numpy(windows)
    w_enc, b_enc = random_encoder_weights(windows_tensor.shape[-1], args.feature_dim, args.seed + 1)
    x0_all = encode_windows(windows_tensor, w_enc, b_enc)
    a_hat = build_channel_graph()
    identity = torch.eye(a_hat.shape[0], dtype=torch.float32)

    omega_all = x0_all.clone()
    omega_identical = omega_all.mean(dim=1, keepdim=True).expand_as(omega_all)

    t_values = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, args.max_T]
    model_names = ["GRAND", "KuramotoGNN nonidentical omega", "KuramotoGNN identical omega"]

    results: list[dict[str, Any]] = []
    metric_history: dict[str, dict[str, list[float]]] = {
        key: {name: [] for name in model_names}
        for key in ("mean_pairwise_distance", "dirichlet_energy", "class_separation", "cosine_similarity")
    }
    state_by_model: dict[str, dict[float, torch.Tensor]] = {name: {} for name in model_names}

    def derivative_for(model_name: str, window_idx: int) -> Callable[[torch.Tensor], torch.Tensor]:
        if "nonidentical" in model_name:
            omega = omega_all[window_idx]
            return lambda x: kuramoto_derivative(x, a_hat, omega, args.K)
        if "identical" in model_name:
            omega = omega_identical[window_idx]
            return lambda x: kuramoto_derivative(x, a_hat, omega, args.K)
        return lambda x: grand_derivative(x, a_hat, identity)

    for model_name in model_names:
        for t_value in t_values:
            averages = {
                key: 0.0
                for key in (
                    "mean_pairwise_distance",
                    "dirichlet_energy",
                    "class_separation",
                    "cosine_similarity",
                    "velocity_sync",
                )
            }
            pooled_states: list[torch.Tensor] = []
            for window_idx in range(x0_all.shape[0]):
                x0 = x0_all[window_idx]
                derivative = derivative_for(model_name, window_idx)
                with torch.no_grad():
                    state = integrate(derivative, x0, t_value, args.dt)
                pooled_states.append(state.mean(dim=0))
                metrics = compute_metrics(state, a_hat, region_labels, derivative if "Kuramoto" in model_name else None)
                for key in averages:
                    averages[key] += metrics[key]
            n_windows = x0_all.shape[0]
            for key in averages:
                averages[key] /= n_windows
            pooled_tensor = torch.stack(pooled_states, dim=0)

            cv_result = train_logistic_classifier_cv(
                pooled_tensor, conditions, fold_indices, seed=args.seed + int(t_value * 100)
            )
            results.append(
                {
                    "model": model_name,
                    "T": t_value,
                    "mean_pairwise_distance": averages["mean_pairwise_distance"],
                    "dirichlet_energy": averages["dirichlet_energy"],
                    "class_separation": averages["class_separation"],
                    "cosine_similarity": averages["cosine_similarity"],
                    "accuracy_mean": cv_result["accuracy_mean"],
                    "accuracy_std": cv_result["accuracy_std"],
                    "auc_mean": cv_result["auc_mean"],
                    "auc_std": cv_result["auc_std"],
                    "overall_accuracy": cv_result["overall_accuracy"],
                    "overall_auc": cv_result["overall_auc"],
                    "fold_metrics": cv_result["fold_metrics"],
                    "confusion_matrix": cv_result["confusion_matrix"],
                }
            )
            if t_value == args.max_T:
                plot_confusion_matrix(
                    cv_result["confusion_matrix"],
                    model_name,
                    t_value,
                    filename=test_output_dir
                    / f"confusion_matrix_{model_name.replace(' ', '_').replace('/', '_')}_T{int(t_value)}.png",
                )
            for key in metric_history:
                metric_history[key][model_name].append(averages[key])
            state_by_model[model_name][t_value] = pooled_tensor

    LOGGER.info(
        "Dataset-based KuramotoGNN versus GRAND metrics (window-level, subject-grouped %d-fold CV):", cv_splits_used
    )
    print_results_table(results)

    plot_metric_curve(
        metric_history["mean_pairwise_distance"],
        t_values,
        "Mean pairwise distance",
        args.output_dir / "mean_pairwise_distance.png",
        "Mean pairwise distance vs terminal time T",
    )
    plot_metric_curve(
        metric_history["dirichlet_energy"],
        t_values,
        "Dirichlet energy",
        args.output_dir / "dirichlet_energy.png",
        "Dirichlet energy vs terminal time T",
    )
    plot_metric_curve(
        metric_history["class_separation"],
        t_values,
        "Class separation score",
        args.output_dir / "class_separation.png",
        "Class separation vs terminal time T",
    )
    plot_metric_curve(
        metric_history["cosine_similarity"],
        t_values,
        "Feature cosine similarity",
        args.output_dir / "cosine_similarity.png",
        "Cosine similarity vs terminal time T",
    )
    plot_pca_representations(
        state_by_model,
        conditions,
        small_t=t_values[1],
        large_t=t_values[-1],
        filename=args.output_dir / "pca_features.png",
    )
    plot_channel_graph(a_hat, filename=args.output_dir / "networkx_graph_visulisation.png")
    plot_connectivity_heatmap(a_hat, filename=args.output_dir / "node_connectivity_heatmap.png")

    summary = build_summary(
        args=args,
        dataset_files=files,
        subjects=subjects,
        window_count=int(len(conditions)),
        cv_splits_used=cv_splits_used,
        results=results,
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Saved figures and summary.json to %s", args.output_dir)


if __name__ == "__main__":
    main()
