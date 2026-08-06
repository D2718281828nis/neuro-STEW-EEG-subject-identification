import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


class KuramotoGNN(nn.Module):
    """A simple GNN that uses graph adjacency and node features.

    The model is inspired by Kuramoto coupling in that nodes influence each other
    through a shared adjacency structure. It is not a true Kuramoto oscillator
    model but uses the same idea of node interactions and synchronization.
    """

    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.lin1 = nn.Linear(in_features, hidden_features)
        self.lin2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # Message passing via adjacency matrix multiplication.
        # This implements one round of node-to-node information exchange.
        x = torch.matmul(adj, x)
        x = self.lin1(x)
        x = F.relu(x)
        x = torch.matmul(adj, x)
        x = self.lin2(x)
        return x


def build_synthetic_kuramoto_graph(num_nodes: int, edge_prob: float = 0.2, seed: int = 42):
    """Builds a synthetic graph with node features and binary labels."""
    rng = np.random.default_rng(seed)
    G = nx.erdos_renyi_graph(n=num_nodes, p=edge_prob, seed=seed)

    # Use node degrees and random phase-like values for features.
    phases = rng.uniform(low=0.0, high=2 * np.pi, size=num_nodes)
    degrees = np.array([deg for _, deg in G.degree()], dtype=float)
    features = np.vstack([np.sin(phases), np.cos(phases), degrees]).T

    # Labels encode whether the node's phase is in the upper half of the circle.
    labels = (phases > np.pi).astype(int)
    nx.set_node_attributes(G, {i: {"feature": features[i], "label": int(labels[i])} for i in G.nodes})
    return G, features.astype(np.float32), labels.astype(np.int64)


def graph_to_adj_matrix(G: nx.Graph) -> np.ndarray:
    """Return a normalized adjacency matrix for GNN propagation."""
    adj = nx.to_numpy_array(G, dtype=np.float32)
    adj = adj + np.eye(adj.shape[0], dtype=np.float32)
    degree = np.sum(adj, axis=1, keepdims=True)
    return adj / degree


def plot_graph(G: nx.Graph, path: str = "graph_visualization.png"):
    """Visualize the network structure using networkx and matplotlib."""
    pos = nx.spring_layout(G, seed=123)
    labels = nx.get_node_attributes(G, "label")
    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color=list(labels.values()), cmap="coolwarm", vmin=0, vmax=1)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=8)
    plt.title("Synthetic Kuramoto-inspired Graph")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_model(model, features_tensor, adj_tensor, labels_tensor, train_idx, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(40):
        optimizer.zero_grad()
        outputs = model(features_tensor, adj_tensor)
        loss = criterion(outputs[train_idx], labels_tensor[train_idx])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1:02d} loss: {loss.item():.4f}")


def evaluate_model(model, features_tensor, adj_tensor, labels_tensor, eval_idx, device):
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor, adj_tensor)
        probabilities = F.softmax(outputs, dim=1)[:, 1]
        true_labels = labels_tensor[eval_idx].cpu().numpy()
        pred_probs = probabilities[eval_idx].cpu().numpy()
    return true_labels, pred_probs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G, features, labels = build_synthetic_kuramoto_graph(num_nodes=100, edge_prob=0.15)
    print("Graph nodes:", G.number_of_nodes(), "edges:", G.number_of_edges())
    plot_graph(G, path="Model KuramotoGNN/kuramoto_graph.png")

    adj_matrix = graph_to_adj_matrix(G)
    adj_tensor = torch.from_numpy(adj_matrix).to(device)
    features_tensor = torch.from_numpy(features).to(device)
    labels_tensor = torch.from_numpy(labels).to(device)

    node_indices = np.arange(features.shape[0])
    train_idx, test_idx = train_test_split(node_indices, test_size=0.3, random_state=42, stratify=labels)
    train_idx = torch.tensor(train_idx, dtype=torch.long, device=device)
    test_idx = torch.tensor(test_idx, dtype=torch.long, device=device)

    model = KuramotoGNN(in_features=3, hidden_features=16, out_features=2).to(device)
    train_model(model, features_tensor, adj_tensor, labels_tensor, train_idx, device)

    y_true, y_scores = evaluate_model(model, features_tensor, adj_tensor, labels_tensor, test_idx, device)
    y_pred = (y_scores >= 0.5).astype(int)
    auc_score = roc_auc_score(y_true, y_scores)
    cm = confusion_matrix(y_true, y_pred)
    print(f"Test ROC-AUC: {auc_score:.4f}")
    print("Confusion matrix:\n", cm)

    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("Model KuramotoGNN/confusion_matrix.png")
    plt.close()

    fig, ax = plt.subplots(figsize=(6, 6))
    RocCurveDisplay.from_predictions(y_true, y_scores, ax=ax)
    ax.set_title("ROC Curve")
    plt.tight_layout()
    plt.savefig("Model KuramotoGNN/roc_curve.png")
    plt.close()

    print(
        "Saved plots to Model KuramotoGNN/kuramoto_graph.png, "
        "Model KuramotoGNN/confusion_matrix.png, Model KuramotoGNN/roc_curve.png"
    )


if __name__ == "__main__":
    main()
