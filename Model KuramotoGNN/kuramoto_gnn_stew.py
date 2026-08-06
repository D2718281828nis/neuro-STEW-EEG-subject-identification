import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

FS = 128
WINDOW_SECONDS = 4
STEP_SECONDS = 2
WINDOW_SAMPLES = FS * WINDOW_SECONDS
STEP_SAMPLES = FS * STEP_SECONDS
CHANNELS = [
    'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1',
    'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4',
]
REGIONS = {
    'PREFRONTAL': ['AF3', 'AF4'],
    'FRONTAL': ['F7', 'F3', 'F4', 'F8'],
    'FRONTOCENTRAL': ['FC5', 'FC6'],
    'TEMPORAL': ['T7', 'T8'],
    'PARIETAL': ['P7', 'P8'],
    'OCCIPITAL': ['O1', 'O2'],
}
BANDS = {
    'theta': (4.0, 8.0),
    'alpha': (8.0, 13.0),
    'beta': (13.0, 30.0),
    'gamma': (30.0, 45.0),
}


@dataclass
class RecordingFeatures:
    subject: int
    condition: int
    node_log_power: np.ndarray


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _channel_indices(names: List[str]) -> List[int]:
    return [CHANNELS.index(name) for name in names]


def _window_view(data: np.ndarray) -> np.ndarray:
    starts = np.arange(0, data.shape[0] - WINDOW_SAMPLES + 1, STEP_SAMPLES)
    return np.stack([data[start:start + WINDOW_SAMPLES] for start in starts], axis=0)


def preprocess_eeg(data: np.ndarray) -> np.ndarray:
    data = data.astype(np.float32)
    data = data - data.mean(axis=0, keepdims=True)
    return data


def relative_band_power(windows: np.ndarray) -> np.ndarray:
    n_windows, n_samples, n_channels = windows.shape
    window_fn = np.hanning(n_samples).astype(np.float32)[None, :, None]
    windows = windows * window_fn
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / FS)
    spectrum = np.fft.rfft(windows, axis=1)
    psd = (np.abs(spectrum) ** 2) / np.sum(window_fn ** 2)
    total_mask = (freqs >= 1.0) & (freqs <= 45.0)
    total = np.sum(psd[:, total_mask, :], axis=1)
    bands = []
    for band_name, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        bands.append(np.sum(psd[:, mask, :], axis=1))
    absolute = np.stack(bands, axis=-1)
    return absolute / np.maximum(total[:, :, None], 1e-12)


def build_node_log_power(relative_power: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(relative_power, 1e-12))


def extract_recording(path: Path) -> RecordingFeatures:
    name = path.name
    subject = int(name[3:5])
    condition = 0 if '_lo' in name else 1
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] != len(CHANNELS):
        raise ValueError(f'Unexpected STEW file shape for {path}: {data.shape}')
    data = preprocess_eeg(data)
    windows = _window_view(data)
    relative_power = relative_band_power(windows)
    node_log_power = build_node_log_power(relative_power)
    return RecordingFeatures(subject=subject, condition=condition, node_log_power=node_log_power)


def baseline_normalize(records: List[RecordingFeatures]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_subject: Dict[int, List[RecordingFeatures]] = {}
    for record in records:
        by_subject.setdefault(record.subject, []).append(record)

    all_windows: List[np.ndarray] = []
    all_conditions: List[int] = []
    for subject, recs in sorted(by_subject.items()):
        rest_windows = np.concatenate([rec.node_log_power for rec in recs if rec.condition == 0], axis=0)
        if rest_windows.shape[0] == 0:
            raise ValueError(f'Missing rest windows for subject {subject}')
        center = np.median(rest_windows, axis=0)
        mad = np.median(np.abs(rest_windows - center), axis=0)
        scale = np.maximum(mad * 1.4826, 1e-6)
        for rec in recs:
            z = (rec.node_log_power - center) / scale
            z[:, :, 1] *= -1.0
            z = np.clip(z, -6.0, 6.0)
            all_windows.append(z.astype(np.float32))
            all_conditions.extend([rec.condition] * z.shape[0])
    return np.concatenate(all_windows, axis=0), np.array(all_conditions, dtype=np.int64), get_channel_region_labels()


def get_channel_region_labels() -> np.ndarray:
    labels = np.zeros(len(CHANNELS), dtype=np.int64)
    region_names = list(REGIONS.keys())
    for region_index, region in enumerate(region_names):
        for channel in REGIONS[region]:
            labels[CHANNELS.index(channel)] = region_index
    return labels


def build_channel_graph() -> torch.Tensor:
    n = len(CHANNELS)
    adj = np.zeros((n, n), dtype=np.float32)
    region_map = {}
    for region, members in REGIONS.items():
        for channel in members:
            region_map[channel] = region
    for i, ci in enumerate(CHANNELS):
        for j, cj in enumerate(CHANNELS):
            if i == j:
                continue
            if region_map[ci] == region_map[cj]:
                adj[i, j] = 1.0
            else:
                adj[i, j] = 0.12
    adj = torch.from_numpy(adj)
    return normalize_row_stochastic(adj)


def normalize_row_stochastic(adj: torch.Tensor) -> torch.Tensor:
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
        raise ValueError('A_hat rows do not sum to 1.')
    return a_hat


def random_encoder_weights(feature_dim: int, hidden_dim: int, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    W = torch.from_numpy(rng.normal(scale=0.5, size=(feature_dim, hidden_dim)).astype(np.float32))
    b = torch.from_numpy(rng.normal(scale=0.1, size=(hidden_dim,)).astype(np.float32))
    return W, b


def encode_windows(windows: torch.Tensor, W: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return windows @ W + b


def integrate(
    derivative_fn: Callable[[torch.Tensor], torch.Tensor],
    X0: torch.Tensor,
    T: float,
    dt: float,
) -> torch.Tensor:
    assert X0.dtype == torch.float32
    assert T >= 0.0 and dt > 0.0
    if T == 0.0:
        return X0.clone()
    X = X0.clone()
    t = 0.0
    while t < T:
        step = min(dt, T - t)
        dXdt = derivative_fn(X)
        X = X + step * dXdt
        if torch.isnan(X).any():
            raise ValueError(f'NaN detected during integration at t={t:.4f}.')
        t += step
    return X


def grand_derivative(X: torch.Tensor, a_hat: torch.Tensor, identity: torch.Tensor) -> torch.Tensor:
    return (a_hat - identity) @ X


def kuramoto_derivative(X: torch.Tensor, a_hat: torch.Tensor, omega: torch.Tensor, K: float) -> torch.Tensor:
    diff = X.unsqueeze(0) - X.unsqueeze(1)
    coupling = torch.einsum('ij,ijd->id', a_hat, torch.sin(diff))
    return omega + K * coupling


def pairwise_distances(X: torch.Tensor) -> torch.Tensor:
    squared_norms = (X * X).sum(dim=1, keepdim=True)
    dist_sq = squared_norms + squared_norms.t() - 2.0 * (X @ X.t())
    dist_sq = torch.clamp(dist_sq, min=0.0)
    return torch.sqrt(dist_sq)


def mean_pairwise_distance(X: torch.Tensor) -> float:
    dist = pairwise_distances(X)
    n = X.shape[0]
    return float(dist.sum() / (n * (n - 1)))


def dirichlet_energy(X: torch.Tensor, a_hat: torch.Tensor) -> float:
    dist_sq = pairwise_distances(X) ** 2
    energy = 0.5 * (a_hat * dist_sq).sum() / X.shape[0]
    return float(energy)


def class_separation_score(X: torch.Tensor, labels: np.ndarray) -> float:
    dist = pairwise_distances(X)
    same = labels[:, None] == labels[None, :]
    diff = ~same
    same = same.astype(np.float32)
    diff = diff.astype(np.float32)
    same_sum = float((dist * torch.from_numpy(same)).sum()) - float(np.trace(dist.numpy()))
    diff_sum = float((dist * torch.from_numpy(diff)).sum())
    same_count = same.sum() - X.shape[0]
    diff_count = diff.sum()
    if same_count <= 0 or diff_count <= 0:
        return 0.0
    return float(diff_sum / diff_count - same_sum / same_count)


def average_cosine_similarity(X: torch.Tensor) -> float:
    normalized = torch.nn.functional.normalize(X, dim=1)
    sim = normalized @ normalized.t()
    n = X.shape[0]
    return float((sim.sum() - n) / (n * (n - 1)))


def velocity_synchronization(X: torch.Tensor, derivative_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:
    V = derivative_fn(X)
    dist = pairwise_distances(V)
    n = V.shape[0]
    return float(dist.sum() / (n * (n - 1)))


def compute_auc(labels: torch.Tensor, scores: torch.Tensor) -> float:
    order = torch.argsort(scores)
    labels = labels[order]
    positives = labels.sum().item()
    negatives = labels.numel() - positives
    if positives == 0 or negatives == 0:
        return float('nan')
    ranks = torch.arange(1, len(labels) + 1, dtype=torch.float32)
    pos_ranks = ranks[labels == 1]
    auc = (pos_ranks.sum() - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def confusion_matrix(labels: torch.Tensor, predictions: torch.Tensor) -> np.ndarray:
    labels = labels.long()
    predictions = predictions.long()
    tn = int(((labels == 0) & (predictions == 0)).sum().item())
    fp = int(((labels == 0) & (predictions == 1)).sum().item())
    fn = int(((labels == 1) & (predictions == 0)).sum().item())
    tp = int(((labels == 1) & (predictions == 1)).sum().item())
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def train_logistic_classifier(features: torch.Tensor, labels: torch.Tensor, seed: int) -> Tuple[float, float, np.ndarray]:
    n_samples, dim = features.shape
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    split = int(0.7 * n_samples)
    train_idx = indices[:split]
    test_idx = indices[split:]
    x_train = features[train_idx]
    y_train = labels[train_idx].float()
    x_test = features[test_idx]
    y_test = labels[test_idx].float()

    w = torch.zeros(dim, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([w, b], lr=0.1)

    for _ in range(200):
        logits = x_train @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_logits = x_test @ w + b
        test_probs = torch.sigmoid(test_logits)
        predictions = (test_probs >= 0.5).float()
        accuracy = float((predictions == y_test).float().mean())
        auc = compute_auc(y_test, test_probs)
        cm = confusion_matrix(y_test, predictions)
    return accuracy, auc, cm


def sample_windows(
    windows: np.ndarray,
    conditions: np.ndarray,
    max_windows: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = windows.shape[0]
    if n <= max_windows:
        return windows, conditions
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rest_idx = indices[conditions == 0]
    high_idx = indices[conditions == 1]
    count = max_windows // 2
    chosen = np.concatenate([
        rng.choice(rest_idx, size=count, replace=False),
        rng.choice(high_idx, size=count, replace=False),
    ])
    rng.shuffle(chosen)
    return windows[chosen], conditions[chosen]


def plot_metric_curve(
    metric_history: Dict[str, List[float]],
    T_values: List[float],
    ylabel: str,
    filename: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    for model_name, values in metric_history.items():
        plt.plot(T_values, values, marker='o', label=model_name)
    plt.xlabel('Terminal time T')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_pca_representations(
    representations: Dict[str, Dict[float, torch.Tensor]],
    labels: np.ndarray,
    small_T: float,
    large_T: float,
    filename: str,
) -> None:
    model_names = list(representations.keys())
    fig, axes = plt.subplots(2, len(model_names), figsize=(5 * len(model_names), 10))
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    X0 = representations[model_names[0]][0.0].cpu().numpy()
    mean = X0.mean(axis=0, keepdims=True)
    cov = np.cov(X0 - mean, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order[:2]]

    for col, model_name in enumerate(model_names):
        for row, T in enumerate((small_T, large_T)):
            X = representations[model_name][T].cpu().numpy()
            X2 = (X - mean) @ components
            ax = axes[row, col]
            scatter = ax.scatter(
                X2[:, 0],
                X2[:, 1],
                c=labels,
                cmap='tab10',
                s=20,
                alpha=0.8,
            )
            ax.set_title(f'{model_name}\nT={T}')
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.grid(True, alpha=0.2)
    fig.suptitle('PCA of pooled window representations by rest/high', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename)
    plt.close()


def plot_channel_graph(adjacency: torch.Tensor, filename: str) -> None:
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
                color='gray', alpha=0.3, linewidth=0.8,
            )
    region_labels = get_channel_region_labels()
    ax.scatter(
        positions[:, 0], positions[:, 1],
        c=region_labels,
        cmap='tab10',
        s=120,
        edgecolor='black',
    )
    for i, channel in enumerate(CHANNELS):
        ax.text(positions[i, 0], positions[i, 1], channel, fontsize=8, ha='center', va='center')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('STEW EEG channel graph for KuramotoGNN')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def plot_connectivity_heatmap(adjacency: torch.Tensor, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(adjacency.cpu().numpy(), cmap='viridis', aspect='equal')
    ax.set_title('Node connectivity heatmap')
    ax.set_xlabel('Target node')
    ax.set_ylabel('Source node')
    ax.set_xticks(np.arange(len(CHANNELS)))
    ax.set_yticks(np.arange(len(CHANNELS)))
    ax.set_xticklabels(CHANNELS, rotation=90, fontsize=8)
    ax.set_yticklabels(CHANNELS, fontsize=8)
    fig.colorbar(im, ax=ax, label='Normalized edge weight')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, model_name: str, T: float, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'Confusion matrix: {model_name} at T={T}')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Rest', 'High'])
    ax.set_yticklabels(['Rest', 'High'])
    fig.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def compute_metrics(
    state: torch.Tensor,
    a_hat: torch.Tensor,
    labels: np.ndarray,
    derivative_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        'mean_pairwise_distance': mean_pairwise_distance(state),
        'dirichlet_energy': dirichlet_energy(state, a_hat),
        'class_separation': class_separation_score(state, labels),
        'cosine_similarity': average_cosine_similarity(state),
        'velocity_sync': float('nan'),
    }
    if derivative_fn is not None:
        metrics['velocity_sync'] = velocity_synchronization(state, derivative_fn)
    return metrics


def print_table(results: List[Dict[str, object]]) -> None:
    header = f"{'Model':<30} {'T':>6} {'MeanDist':>10} {'Dirichlet':>12} {'ClassSep':>10} {'CosSim':>9} {'Acc':>7} {'AUC':>7}"
    print(header)
    print('-' * len(header))
    for row in results:
        print(
            f"{row['model']:<30} {row['T']:6.1f} {row['mean_pairwise_distance']:10.4f} "
            f"{row['dirichlet_energy']:12.4f} {row['class_separation']:10.4f} {row['cosine_similarity']:9.4f} "
            f"{row['accuracy']:7.3f} {row['auc']:7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='KuramotoGNN content on STEW EEG dataset')
    parser.add_argument('--dataset', type=Path, default=Path('dataset'))
    parser.add_argument('--sample_windows', type=int, default=300)
    parser.add_argument('--feature_dim', type=int, default=20)
    parser.add_argument('--K', type=float, default=2.0)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--max_T', type=float, default=32.0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--output_dir', type=Path, default=Path('Model KuramotoGNN'))
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    test_output_dir = args.output_dir / 'test_output'
    test_output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(args.dataset.glob('sub??_*.txt'))
    if len(files) == 0:
        raise ValueError(f'No STEW files found in {args.dataset}')
    records = [extract_recording(path) for path in files]
    windows, conditions, region_labels = baseline_normalize(records)
    windows, conditions = sample_windows(windows, conditions, args.sample_windows, args.seed)

    windows_tensor = torch.from_numpy(windows)
    W, b = random_encoder_weights(windows_tensor.shape[-1], args.feature_dim, args.seed + 1)
    X0_all = encode_windows(windows_tensor, W, b)
    a_hat = build_channel_graph()
    identity = torch.eye(a_hat.shape[0], dtype=torch.float32)

    omega_all = X0_all.clone()
    omega_identical = omega_all.mean(dim=1, keepdim=True).expand_as(omega_all)

    T_values = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, args.max_T]
    model_specs = [
        ('GRAND', lambda X: grand_derivative(X, a_hat, identity), None),
        ('KuramotoGNN nonidentical omega', lambda X: kuramoto_derivative(X, a_hat, omega_all[0], args.K), lambda X: kuramoto_derivative(X, a_hat, omega_all[0], args.K)),
        ('KuramotoGNN identical omega', lambda X: kuramoto_derivative(X, a_hat, omega_identical[0], args.K), lambda X: kuramoto_derivative(X, a_hat, omega_identical[0], args.K)),
    ]

    # We must use the same base graph and random encoder for all windows; omega differs per window instance.
    results: List[Dict[str, object]] = []
    metric_history: Dict[str, Dict[str, List[float]]] = {
        'mean_pairwise_distance': {},
        'dirichlet_energy': {},
        'class_separation': {},
        'cosine_similarity': {},
    }
    state_by_model: Dict[str, Dict[float, torch.Tensor]] = {name: {} for name, _, _ in model_specs}

    # Precompute the region labels for node-level class separation.
    node_labels = region_labels
    pooled_by_model: Dict[str, Dict[float, List[torch.Tensor]]] = {name: {} for name, _, _ in model_specs}

    for model_name, derivative_fn, _ in model_specs:
        for key in metric_history:
            metric_history[key][model_name] = []
        for T in T_values:
            averages = {'mean_pairwise_distance': 0.0, 'dirichlet_energy': 0.0, 'class_separation': 0.0, 'cosine_similarity': 0.0, 'velocity_sync': 0.0}
            pooled_states: List[torch.Tensor] = []
            for window_idx in range(X0_all.shape[0]):
                X0 = X0_all[window_idx]
                omega = omega_all[window_idx]
                if 'nonidentical' in model_name:
                    derivative = lambda X, omega=omega: kuramoto_derivative(X, a_hat, omega, args.K)
                elif 'identical' in model_name:
                    derivative = lambda X, omega=omega_identical[window_idx]: kuramoto_derivative(X, a_hat, omega, args.K)
                else:
                    derivative = lambda X: grand_derivative(X, a_hat, identity)
                with torch.no_grad():
                    state = integrate(derivative, X0, T, args.dt)
                pooled_states.append(state.mean(dim=0))
                metrics = compute_metrics(state, a_hat, node_labels, derivative if 'Kuramoto' in model_name else None)
                averages['mean_pairwise_distance'] += metrics['mean_pairwise_distance']
                averages['dirichlet_energy'] += metrics['dirichlet_energy']
                averages['class_separation'] += metrics['class_separation']
                averages['cosine_similarity'] += metrics['cosine_similarity']
                averages['velocity_sync'] += metrics['velocity_sync']
            n_windows = X0_all.shape[0]
            for key in averages:
                averages[key] /= n_windows
            pooled_tensor = torch.stack(pooled_states, dim=0)
            labels_tensor = torch.from_numpy(conditions)
            accuracy, auc, cm = train_logistic_classifier(pooled_tensor, labels_tensor, args.seed + int(T * 100))
            results.append({
                'model': model_name,
                'T': T,
                'mean_pairwise_distance': averages['mean_pairwise_distance'],
                'dirichlet_energy': averages['dirichlet_energy'],
                'class_separation': averages['class_separation'],
                'cosine_similarity': averages['cosine_similarity'],
                'accuracy': accuracy,
                'auc': auc,
                'confusion_matrix': cm,
            })
            if T == args.max_T:
                plot_confusion_matrix(
                    cm,
                    model_name,
                    T,
                    filename=test_output_dir / f"confusion_matrix_{model_name.replace(' ', '_').replace('/', '_')}_T{int(T)}.png",
                )
            metric_history['mean_pairwise_distance'][model_name].append(averages['mean_pairwise_distance'])
            metric_history['dirichlet_energy'][model_name].append(averages['dirichlet_energy'])
            metric_history['class_separation'][model_name].append(averages['class_separation'])
            metric_history['cosine_similarity'][model_name].append(averages['cosine_similarity'])
            state_by_model[model_name][T] = pooled_tensor

    print('\nDataset-based KuramotoGNN versus GRAND metrics:')
    print_table(results)

    plot_metric_curve(
        metric_history['mean_pairwise_distance'],
        T_values,
        ylabel='Mean pairwise distance',
        filename=args.output_dir / 'mean_pairwise_distance.png',
        title='Mean pairwise distance vs terminal time T',
    )
    plot_metric_curve(
        metric_history['dirichlet_energy'],
        T_values,
        ylabel='Dirichlet energy',
        filename=args.output_dir / 'dirichlet_energy.png',
        title='Dirichlet energy vs terminal time T',
    )
    plot_metric_curve(
        metric_history['class_separation'],
        T_values,
        ylabel='Class separation score',
        filename=args.output_dir / 'class_separation.png',
        title='Class separation vs terminal time T',
    )
    plot_metric_curve(
        metric_history['cosine_similarity'],
        T_values,
        ylabel='Feature cosine similarity',
        filename=args.output_dir / 'cosine_similarity.png',
        title='Cosine similarity vs terminal time T',
    )
    plot_pca_representations(
        state_by_model,
        conditions,
        small_T=T_values[1],
        large_T=T_values[-1],
        filename=args.output_dir / 'pca_features.png',
    )
    plot_channel_graph(a_hat, filename=args.output_dir / 'networkx_graph_visulisation.png')
    plot_connectivity_heatmap(a_hat, filename=args.output_dir / 'node_connectivity_heatmap.png')

    print('\nSaved figures to', args.output_dir)


if __name__ == '__main__':
    main()
