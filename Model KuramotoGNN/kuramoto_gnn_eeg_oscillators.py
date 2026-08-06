import argparse
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_sbm_graph(
    num_nodes: int,
    num_classes: int,
    within_prob: float,
    between_prob: float,
    seed: int,
) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Build a stochastic block model adjacency matrix for EEG contact nodes."""
    rng = np.random.default_rng(seed)
    class_sizes = np.full(num_classes, num_nodes // num_classes, dtype=int)
    remainder = num_nodes - class_sizes.sum()
    class_sizes[:remainder] += 1

    labels = np.repeat(np.arange(num_classes), class_sizes)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = within_prob if labels[i] == labels[j] else between_prob
            if rng.random() < p:
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0

    np.fill_diagonal(adjacency, 0.0)

    # Ensure every node has at least one connection.
    for node in range(num_nodes):
        if adjacency[node].sum() == 0.0:
            same_class = np.where(labels == labels[node])[0]
            neighbor = rng.choice(same_class[same_class != node])
            adjacency[node, neighbor] = 1.0
            adjacency[neighbor, node] = 1.0

    adjacency = torch.from_numpy(adjacency)
    return adjacency, labels, class_sizes


def normalize_row_stochastic(adj: torch.Tensor) -> torch.Tensor:
    """Convert adjacency to a row-stochastic matrix A_hat with rows summing to 1."""
    assert adj.dim() == 2 and adj.shape[0] == adj.shape[1]
    row_sum = adj.sum(dim=1, keepdim=True)
    zero_rows = row_sum.squeeze(1) == 0.0
    safe_adj = adj.clone()
    if zero_rows.any():
        safe_adj[zero_rows, :] = 0.0
        safe_adj[zero_rows, torch.arange(adj.shape[0])[zero_rows]] = 1.0
        row_sum = safe_adj.sum(dim=1, keepdim=True)

    a_hat = safe_adj / row_sum
    row_sum_after = a_hat.sum(dim=1)
    if not torch.allclose(row_sum_after, torch.ones_like(row_sum_after), atol=1e-5):
        raise ValueError('A_hat rows do not sum to 1 after normalization.')
    return a_hat


def build_eeg_like_features(
    num_nodes: int,
    feature_dim: int,
    labels: np.ndarray,
    num_classes: int,
    seed: int,
) -> torch.Tensor:
    """Generate synthetic EEG contact features with class-level rhythm patterns."""
    rng = np.random.default_rng(seed)
    rhythm_frequencies = np.array([10.0, 20.0, 6.0])  # alpha, beta, theta Hz-like bands
    freq_grid = np.linspace(0.0, 1.0, feature_dim)
    class_means = np.zeros((num_classes, feature_dim), dtype=np.float32)

    for c in range(num_classes):
        base = np.sin(2.0 * np.pi * rhythm_frequencies[c] * freq_grid / feature_dim)
        class_means[c] = base + rng.normal(scale=0.05, size=feature_dim)

    features = class_means[labels] + rng.normal(scale=0.2, size=(num_nodes, feature_dim))
    features = torch.from_numpy(features.astype(np.float32))
    features = torch.nn.functional.normalize(features, dim=1)
    return features


def random_encoder(features: torch.Tensor, seed: int) -> torch.Tensor:
    """Encode raw EEG-like features into initial node states X0."""
    rng = np.random.default_rng(seed)
    num_features = features.shape[1]
    hidden_dim = num_features
    W = torch.from_numpy(rng.normal(scale=0.5, size=(num_features, hidden_dim)).astype(np.float32))
    b = torch.from_numpy(rng.normal(scale=0.1, size=(hidden_dim,)).astype(np.float32))
    return features @ W + b


class MLPClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def integrate(
    derivative_fn: Callable[[torch.Tensor], torch.Tensor],
    X0: torch.Tensor,
    T: float,
    dt: float,
) -> torch.Tensor:
    """Integrate ordinary differential dynamics using explicit Euler."""
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


def grand_derivative(
    X: torch.Tensor,
    a_hat: torch.Tensor,
    identity: torch.Tensor,
    residual: torch.Tensor,
    beta: float = 0.5,
) -> torch.Tensor:
    """Residual GRAND diffusion with an attractor to the initial embedding.

    Pure GRAND diffusion using (A_hat - I) tends to oversmooth node states toward
    a consensus solution, which can destroy class-specific structure. The residual
    attractor encourages X to retain its initial encoding while also diffusing
    information across the graph.
    """
    diffusion = (a_hat - identity) @ X
    attractor = residual - X
    return beta * diffusion + (1.0 - beta) * attractor


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
    mean_val = dist.sum() / (n * (n - 1))
    return float(mean_val)


def dirichlet_energy(X: torch.Tensor, a_hat: torch.Tensor) -> float:
    dist_sq = pairwise_distances(X) ** 2
    energy = 0.5 * (a_hat * dist_sq).sum() / X.shape[0]
    return float(energy)


def class_separation_score(X: torch.Tensor, labels: np.ndarray) -> float:
    n = X.shape[0]
    dist = pairwise_distances(X)
    same_mask = labels[:, None] == labels[None, :]
    diff_mask = ~same_mask
    same_mask = same_mask.astype(np.float32)
    diff_mask = diff_mask.astype(np.float32)
    same_sum = float((dist * torch.from_numpy(same_mask)).sum())
    diff_sum = float((dist * torch.from_numpy(diff_mask)).sum())
    same_count = same_mask.sum() - n
    diff_count = diff_mask.sum()
    if same_count <= 0 or diff_count <= 0:
        return 0.0
    same_mean = same_sum / same_count
    diff_mean = diff_sum / diff_count
    return float(diff_mean - same_mean)


def average_cosine_similarity(X: torch.Tensor) -> float:
    normalized = torch.nn.functional.normalize(X, dim=1)
    sim = normalized @ normalized.t()
    n = X.shape[0]
    total_sim = sim.sum() - n
    return float(total_sim / (n * (n - 1)))


def velocity_synchronization(X: torch.Tensor, derivative_fn: Callable[[torch.Tensor], torch.Tensor]) -> float:
    V = derivative_fn(X)
    dist = pairwise_distances(V)
    n = V.shape[0]
    mean_val = dist.sum() / (n * (n - 1))
    return float(mean_val)


def fit_pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    centered = X - mean
    cov = centered.T @ centered / (centered.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    components = eigvecs[:, order[:n_components]]
    return mean, components


def transform_pca(X: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (X - mean) @ components


def plot_metric_curve(
    results: Dict[str, List[float]],
    T_values: List[float],
    ylabel: str,
    filename: str,
    title: str,
) -> None:
    plt.figure(figsize=(8, 5))
    for model_name, values in results.items():
        plt.plot(T_values, values, marker='o', label=model_name)
    plt.xlabel('Terminal time T')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_feature_pca(
    state_by_model: Dict[str, Dict[float, torch.Tensor]],
    labels: np.ndarray,
    small_T: float,
    large_T: float,
    filename: str,
) -> None:
    model_names = list(state_by_model.keys())
    n_models = len(model_names)
    X0 = state_by_model[model_names[0]][0.0].cpu().numpy()
    mean, components = fit_pca(X0, n_components=2)

    fig, axes = plt.subplots(2, n_models, figsize=(4 * n_models, 8))
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]

    for col, model_name in enumerate(model_names):
        for row, T in enumerate((small_T, large_T)):
            X = state_by_model[model_name][T].cpu().numpy()
            X2 = transform_pca(X, mean, components)
            ax = axes[row, col]
            scatter = ax.scatter(
                X2[:, 0],
                X2[:, 1],
                c=labels,
                cmap='tab10',
                s=30,
                alpha=0.8,
            )
            ax.set_title(f'{model_name}\nT={T}')
            ax.set_xlabel('PCA 1')
            ax.set_ylabel('PCA 2')
            ax.grid(True, alpha=0.2)

    fig.suptitle('PCA of node representations at small and large T', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(filename)
    plt.close()


def plot_training_history(history: Dict[str, List[float]], filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='train')
    axes[0].plot(history['val_loss'], label='val')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_accuracy'], label='train')
    axes[1].plot(history['val_accuracy'], label='val')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, model_name: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.set_title(f'Confusion matrix: {model_name}')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels([0, 1, 2])
    ax.set_yticklabels([0, 1, 2])
    fig.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > thresh else 'black')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def train_classifier(
    X: torch.Tensor,
    y: torch.Tensor,
    num_classes: int,
    seed: int,
    train_frac: float = 0.8,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> Tuple[MLPClassifier, Dict[str, List[float]], np.ndarray]:
    torch.manual_seed(seed)
    n = X.shape[0]
    indices = torch.randperm(n)
    split = int(train_frac * n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    x_train = X[train_idx]
    y_train = y[train_idx]
    x_val = X[val_idx]
    y_val = y[val_idx]

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MLPClassifier(X.shape[1], hidden_dim=64, num_classes=num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_accuracy': [],
        'val_accuracy': [],
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
            train_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        train_loss /= len(train_dataset)
        train_acc = train_correct / len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
        val_loss /= len(val_dataset)
        val_acc = val_correct / len(val_dataset)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

    with torch.no_grad():
        logits = model(x_val)
        preds = logits.argmax(dim=1)
        cm = confusion_matrix_matrix(y_val, preds)

    return model, history, cm


def confusion_matrix_matrix(y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
    y_true = y_true.long()
    y_pred = y_pred.long()
    num_classes = int(max(y_true.max().item(), y_pred.max().item()) + 1)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def plot_graph(adjacency: torch.Tensor, labels: np.ndarray, filename: str) -> None:
    n = adjacency.shape[0]
    num_classes = int(labels.max() + 1)
    class_centers = np.stack(
        [np.cos(np.linspace(0, 2 * np.pi, num_classes, endpoint=False)),
         np.sin(np.linspace(0, 2 * np.pi, num_classes, endpoint=False))],
        axis=1,
    )
    positions = np.zeros((n, 2), dtype=np.float32)
    for c in range(num_classes):
        class_nodes = np.where(labels == c)[0]
        theta = np.linspace(0, 2 * np.pi, len(class_nodes), endpoint=False)
        positions[class_nodes, 0] = class_centers[c, 0] + 0.6 * np.cos(theta)
        positions[class_nodes, 1] = class_centers[c, 1] + 0.6 * np.sin(theta)

    fig, ax = plt.subplots(figsize=(8, 8))
    edge_indices = torch.nonzero(adjacency > 0.5, as_tuple=False).cpu().numpy()
    for i, j in edge_indices:
        if i < j:
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color='gray',
                alpha=0.3,
                linewidth=0.8,
            )
    scatter = ax.scatter(
        positions[:, 0], positions[:, 1],
        c=labels,
        cmap='tab10',
        s=80,
        edgecolor='k',
        linewidth=0.4,
    )
    ax.set_title('Synthetic SBM EEG contact graph (class-colored)')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.legend(*scatter.legend_elements(), title='Class', loc='upper right')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close()


def compute_metrics(
    state: torch.Tensor,
    a_hat: torch.Tensor,
    labels: np.ndarray,
    model_name: str,
    derivative_fn: Callable[[torch.Tensor], torch.Tensor] = None,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    metrics['mean_pairwise_distance'] = mean_pairwise_distance(state)
    metrics['dirichlet_energy'] = dirichlet_energy(state, a_hat)
    metrics['class_separation'] = class_separation_score(state, labels)
    metrics['cosine_similarity'] = average_cosine_similarity(state)
    if derivative_fn is not None:
        metrics['velocity_sync'] = velocity_synchronization(state, derivative_fn)
    else:
        metrics['velocity_sync'] = float('nan')
    return metrics


def print_table(results: List[Dict[str, object]]) -> None:
    header = f"{'Model':<30} {'T':>6} {'MeanDist':>10} {'Dirichlet':>12} {'ClassSep':>10} {'CosSim':>9}"
    print(header)
    print('-' * len(header))
    for row in results:
        print(
            f"{row['model']:<30} {row['T']:6.1f} {row['mean_pairwise_distance']:10.4f} "
            f"{row['dirichlet_energy']:12.4f} {row['class_separation']:10.4f} {row['cosine_similarity']:9.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description='KuramotoGNN vs GRAND diffusion demonstration')
    parser.add_argument('--num_nodes', type=int, default=300)
    parser.add_argument('--feature_dim', type=int, default=20)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--K', type=float, default=2.0)
    parser.add_argument('--dt', type=float, default=0.05)
    parser.add_argument('--max_T', type=float, default=32.0)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--within_prob', type=float, default=0.15)
    parser.add_argument('--between_prob', type=float, default=0.01)
    parser.add_argument('--grand_beta', type=float, default=0.5,
                        help='Residual strength for GRAND diffusion (lower preserves more original features)')
    args = parser.parse_args()

    set_seed(args.seed)

    adjacency, labels, _ = build_sbm_graph(
        num_nodes=args.num_nodes,
        num_classes=args.num_classes,
        within_prob=args.within_prob,
        between_prob=args.between_prob,
        seed=args.seed,
    )

    features = build_eeg_like_features(
        num_nodes=args.num_nodes,
        feature_dim=args.feature_dim,
        labels=labels,
        num_classes=args.num_classes,
        seed=args.seed + 1,
    )

    A_hat = normalize_row_stochastic(adjacency)
    identity = torch.eye(args.num_nodes, dtype=torch.float32)

    X0 = random_encoder(features, seed=args.seed + 2)
    omega = X0.clone()
    omega_identical = omega.mean(dim=0, keepdim=True).expand_as(omega)

    T_values = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, args.max_T]
    models = [
        ('GRAND', lambda X: grand_derivative(X, A_hat, identity, X0, beta=args.grand_beta), None),
        (
            'KuramotoGNN nonidentical omega',
            lambda X: kuramoto_derivative(X, A_hat, omega, args.K),
            lambda X: kuramoto_derivative(X, A_hat, omega, args.K),
        ),
        (
            'KuramotoGNN identical omega',
            lambda X: kuramoto_derivative(X, A_hat, omega_identical, args.K),
            lambda X: kuramoto_derivative(X, A_hat, omega_identical, args.K),
        ),
    ]

    results: List[Dict[str, object]] = []
    metric_history: Dict[str, Dict[str, List[float]]] = {
        'mean_pairwise_distance': {},
        'dirichlet_energy': {},
        'class_separation': {},
        'cosine_similarity': {},
    }
    state_by_model: Dict[str, Dict[float, torch.Tensor]] = {name: {} for name, _, _ in models}

    for model_name, derivative_fn, velocity_fn in models:
        metric_history['mean_pairwise_distance'][model_name] = []
        metric_history['dirichlet_energy'][model_name] = []
        metric_history['class_separation'][model_name] = []
        metric_history['cosine_similarity'][model_name] = []

        for T in T_values:
            with torch.no_grad():
                state = integrate(lambda X: derivative_fn(X), X0, T, args.dt)
            state_by_model[model_name][T] = state
            metrics = compute_metrics(state, A_hat, labels, model_name, velocity_fn)
            results.append(
                {
                    'model': model_name,
                    'T': T,
                    'mean_pairwise_distance': metrics['mean_pairwise_distance'],
                    'dirichlet_energy': metrics['dirichlet_energy'],
                    'class_separation': metrics['class_separation'],
                    'cosine_similarity': metrics['cosine_similarity'],
                }
            )
            metric_history['mean_pairwise_distance'][model_name].append(metrics['mean_pairwise_distance'])
            metric_history['dirichlet_energy'][model_name].append(metrics['dirichlet_energy'])
            metric_history['class_separation'][model_name].append(metrics['class_separation'])
            metric_history['cosine_similarity'][model_name].append(metrics['cosine_similarity'])

    print('\nObserved metric trends for each model and T:')
    print_table(results)

    plot_metric_curve(
        metric_history['mean_pairwise_distance'],
        T_values,
        ylabel='Mean pairwise distance',
        filename='Model KuramotoGNN/mean_pairwise_distance.png',
        title='Mean pairwise distance vs terminal time T',
    )

    plot_metric_curve(
        metric_history['dirichlet_energy'],
        T_values,
        ylabel='Dirichlet energy',
        filename='Model KuramotoGNN/dirichlet_energy.png',
        title='Dirichlet energy vs terminal time T',
    )

    plot_metric_curve(
        metric_history['class_separation'],
        T_values,
        ylabel='Class separation score',
        filename='Model KuramotoGNN/class_separation.png',
        title='Class separation vs terminal time T',
    )

    plot_metric_curve(
        metric_history['cosine_similarity'],
        T_values,
        ylabel='Feature cosine similarity',
        filename='Model KuramotoGNN/cosine_similarity.png',
        title='Cosine similarity vs terminal time T',
    )

    plot_feature_pca(
        state_by_model,
        labels,
        small_T=T_values[1] if len(T_values) > 1 else T_values[0],
        large_T=T_values[-1],
        filename='Model KuramotoGNN/pca_features.png',
    )

    plot_graph(adjacency, labels, filename='Model KuramotoGNN/networkx_graph_visulisation.png')

    # Train classifiers on final state representations and plot training history.
    for model_name in state_by_model:
        final_state = state_by_model[model_name][args.max_T]
        classifier, history, cm = train_classifier(
            final_state,
            torch.from_numpy(labels).long(),
            num_classes=args.num_classes,
            seed=args.seed,
            epochs=150,
            lr=1e-3,
            batch_size=32,
        )
        hist_filename = f"Model KuramotoGNN/{model_name.replace(' ', '_')}_training_history.png"
        cm_filename = f"Model KuramotoGNN/{model_name.replace(' ', '_')}_confusion_matrix.png"
        plot_training_history(history, hist_filename)
        plot_confusion_matrix(cm, model_name, cm_filename)
        print(f"\nClassifier results for {model_name} at T={args.max_T}:")
        print(f"  Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
        print(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
        print(f"  Training history saved to: {hist_filename}")
        print(f"  Confusion matrix saved to: {cm_filename}")

    print('\nSaved figures:')
    print('- Model KuramotoGNN/mean_pairwise_distance.png')
    print('- Model KuramotoGNN/dirichlet_energy.png')
    print('- Model KuramotoGNN/class_separation.png')
    print('- Model KuramotoGNN/cosine_similarity.png')
    print('- Model KuramotoGNN/pca_features.png')
    print('- Model KuramotoGNN/networkx_graph_visulisation.png')
    for model_name in state_by_model:
        print(f"- Model KuramotoGNN/{model_name.replace(' ', '_')}_training_history.png")
        print(f"- Model KuramotoGNN/{model_name.replace(' ', '_')}_confusion_matrix.png")


if __name__ == '__main__':
    main()
