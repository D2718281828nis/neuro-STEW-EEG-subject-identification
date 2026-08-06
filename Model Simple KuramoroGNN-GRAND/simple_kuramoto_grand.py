#!/usr/bin/env python3
"""
Simple KuramoroGNN-GRAND: Educational Graph Dynamics Demonstration

Demonstrates Kuramoto oscillators and GRAND diffusion on a 3-node graph.
Usage: python "Model Simple KuramoroGNN-GRAND/simple_kuramoto_grand.py"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class GraphConfig:
    adjacency: np.ndarray = field(default_factory=lambda: np.array([[0.0, 1.0, 0.2], [1.0, 0.0, 0.6], [0.2, 0.6, 0.0]], dtype=float))
    node_names: Tuple[str, str, str] = ("A", "B", "C")

@dataclass(frozen=True)
class KuramotoConfig:
    initial_phases: np.ndarray = field(default_factory=lambda: np.array([0.0, 2.0, 4.0], dtype=float))
    natural_frequencies: np.ndarray = field(default_factory=lambda: np.array([1.00, 1.12, 0.90], dtype=float))
    coupling_strength: float = 2.0
    dt: float = 0.01
    duration: float = 10.0

@dataclass(frozen=True)
class GrandConfig:
    initial_features: np.ndarray = field(default_factory=lambda: np.array([1.0, -0.5, 0.25], dtype=float))
    dt: float = 0.01
    duration: float = 10.0

@dataclass
class KuramotoResult:
    times: np.ndarray
    phases: np.ndarray
    phase_velocities: np.ndarray
    message_matrices: np.ndarray
    coupling_terms: np.ndarray
    order_parameters: np.ndarray
    config: KuramotoConfig

@dataclass
class GrandResult:
    times: np.ndarray
    features: np.ndarray
    feature_derivatives: np.ndarray
    message_matrices: np.ndarray
    variances: np.ndarray
    dirichlet_energies: np.ndarray
    normalized_adjacency: np.ndarray
    config: GrandConfig

# =============================================================================
# VALIDATION
# =============================================================================

def validate_adjacency(adj: np.ndarray, node_names: Tuple[str, ...] | None = None) -> None:
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got {adj.shape}")
    n = adj.shape[0]
    if node_names is not None and len(node_names) != n:
        raise ValueError(f"Node names length {len(node_names)} != matrix size {n}")
    if not np.all(np.isfinite(adj)):
        raise ValueError("Adjacency contains non-finite values") 
    if not np.allclose(adj, adj.T):
        raise ValueError("Adjacency must be symmetric for undirected graph")
    if not np.all(adj >= 0):
        raise ValueError("Adjacency weights must be nonnegative")
    if not np.allclose(np.diag(adj), 0):
        raise ValueError("Adjacency diagonal must be zero (no self-loops)")

def validate_time_parameters(dt: float, duration: float) -> None:
    if dt <= 0: raise ValueError(f"dt must be positive, got {dt}")
    if duration < 0: raise ValueError(f"duration must be nonnegative, got {duration}")
    if not (math.isfinite(dt) and math.isfinite(duration)):
        raise ValueError("dt and duration must be finite")

# =============================================================================
# GRAPH UTILITIES  
# =============================================================================

def compute_degree_matrix(adj: np.ndarray) -> np.ndarray:
    return np.diag(np.sum(adj, axis=1))

def normalize_adjacency(adj: np.ndarray, method: Literal["symmetric", "row"] = "symmetric") -> np.ndarray:
    n = adj.shape[0]
    deg = np.diag(np.sum(adj, axis=1))
    if method == "symmetric":
        deg_inv_sqrt = np.array([1.0/np.sqrt(deg[i,i]) if deg[i,i] > 0 else 0.0 for i in range(n)])
        return np.diag(deg_inv_sqrt) @ adj @ np.diag(deg_inv_sqrt)
    else:
        deg_inv = np.array([1.0/deg[i,i] if deg[i,i] > 0 else 0.0 for i in range(n)])
        return np.diag(deg_inv) @ adj

def compute_laplacian(adj: np.ndarray, normalized: bool = False) -> np.ndarray:
    deg = compute_degree_matrix(adj)
    if normalized:
        return np.eye(adj.shape[0]) - normalize_adjacency(adj, "symmetric")
    return deg - adj

# =============================================================================
# KURAMOTO FUNCTIONS
# =============================================================================

def kuramoto_message_matrix(phases: np.ndarray, adj: np.ndarray, K: float) -> np.ndarray:
    # messages[i,j] = K * A[i,j] * sin(theta_j - theta_i) - message from j to i
    phase_diff = phases[np.newaxis, :] - phases[:, np.newaxis]  # (3,3): diff[i,j] = theta_j - theta_i
    return K * adj * np.sin(phase_diff)

def kuramoto_derivative(phases: np.ndarray, omega: np.ndarray, adj: np.ndarray, K: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    messages = kuramoto_message_matrix(phases, adj, K)
    coupling_terms = np.sum(messages, axis=1)  # Sum over j
    return omega + coupling_terms, messages, coupling_terms

def kuramoto_order_parameter(phases: np.ndarray) -> float | np.ndarray:
    if phases.ndim == 1:
        return float(np.abs(np.mean(np.exp(1j * phases))))
    return np.abs(np.sum(np.exp(1j * phases), axis=1)) / phases.shape[1]

def wrapped_phase_difference(diff: float, convention: Literal["pi", "2pi"] = "pi") -> float:
    return float((diff + np.pi) % (2 * np.pi) - np.pi) if convention == "pi" else float(diff % (2 * np.pi))

# =============================================================================
# GRAND FUNCTIONS
# =============================================================================

def grand_message_matrix(features: np.ndarray, adj_norm: np.ndarray) -> np.ndarray:
    return adj_norm * features[np.newaxis, :]  # messages[i,j] = A_norm[i,j] * X_j - message from j to i

def grand_derivative(features: np.ndarray, adj_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    messages = grand_message_matrix(features, adj_norm)
    aggregated = np.sum(messages, axis=1)
    return aggregated - features, messages, aggregated  # dX/dt = (A_norm - I) * X

def dirichlet_energy(features: np.ndarray, adj: np.ndarray) -> float:
    energy = 0.0
    n = features.shape[0]
    for i in range(n):
        for j in range(n):
            energy += adj[i, j] * (features[i] - features[j]) ** 2
    return float(energy)

# =============================================================================
# NUMERICAL INTEGRATION
# =============================================================================

def euler_integrate(func: Callable, initial: np.ndarray, dt: float, duration: float, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    validate_time_parameters(dt, duration)
    n_steps = int(duration / dt) + 1
    times = np.linspace(0, duration, n_steps)
    trajectory = np.zeros((n_steps, initial.shape[0]))
    trajectory[0] = initial
    current = initial.copy()
    for step in range(1, n_steps):
        deriv = func(current, *args, **kwargs)
        if not np.all(np.isfinite(deriv)): raise RuntimeError(f"Non-finite derivative at t={times[step-1]}")
        current = current + dt * deriv
        if not np.all(np.isfinite(current)): raise RuntimeError(f"Non-finite state at t={times[step]}")
        trajectory[step] = current
    return times, trajectory

def rk4_integrate(func: Callable, initial: np.ndarray, dt: float, duration: float, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    validate_time_parameters(dt, duration)
    n_steps = int(duration / dt) + 1
    times = np.linspace(0, duration, n_steps)
    trajectory = np.zeros((n_steps, initial.shape[0]))
    trajectory[0] = initial
    current = initial.copy()
    for step in range(1, n_steps):
        k1 = func(current, *args, **kwargs)
        k2 = func(current + 0.5 * dt * k1, *args, **kwargs)
        k3 = func(current + 0.5 * dt * k2, *args, **kwargs)  
        k4 = func(current + dt * k3, *args, **kwargs)
        for k in [k1, k2, k3, k4]:
            if not np.all(np.isfinite(k)): raise RuntimeError("Non-finite RK4 stage")
        current = current + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.all(np.isfinite(current)): raise RuntimeError("Non-finite state")
        trajectory[step] = current
    return times, trajectory

# =============================================================================
# SIMULATION
# =============================================================================

def simulate_kuramoto(config: KuramotoConfig, graph: GraphConfig, integrator: str = "euler") -> KuramotoResult:
    logger.info("Starting Kuramoto simulation")
    validate_adjacency(graph.adjacency, graph.node_names)
    validate_time_parameters(config.dt, config.duration)
    n = graph.adjacency.shape[0]
    if len(config.initial_phases) != n or len(config.natural_frequencies) != n:
        raise ValueError(f"Config dimensions don't match graph size {n}")
    
    def deriv(phases):
        d, _, _ = kuramoto_derivative(phases, config.natural_frequencies, graph.adjacency, config.coupling_strength)
        return d
    
    if integrator == "euler":
        times, phases = euler_integrate(deriv, config.initial_phases, config.dt, config.duration)
    else:
        times, phases = rk4_integrate(deriv, config.initial_phases, config.dt, config.duration)
    
    n_times = len(times)
    phase_vels = np.zeros((n_times, n))
    msg_matrices = np.zeros((n_times, n, n))
    coupling_terms = np.zeros((n_times, n)) 
    order_params = np.zeros(n_times)
    
    for t_idx in range(n_times):
        p = phases[t_idx]
        d, m, c = kuramoto_derivative(p, config.natural_frequencies, graph.adjacency, config.coupling_strength)
        phase_vels[t_idx] = d
        msg_matrices[t_idx] = m
        coupling_terms[t_idx] = c
        order_params[t_idx] = kuramoto_order_parameter(p)
    
    return KuramotoResult(times, phases, phase_vels, msg_matrices, coupling_terms, order_params, config)

def simulate_grand(config: GrandConfig, graph: GraphConfig, integrator: str = "euler", method: str = "symmetric") -> GrandResult:
    logger.info("Starting GRAND simulation")
    validate_adjacency(graph.adjacency, graph.node_names)
    validate_time_parameters(config.dt, config.duration)
    n = graph.adjacency.shape[0]
    if len(config.initial_features) != n:
        raise ValueError(f"Config dimensions don't match graph size {n}")
    
    adj_norm = normalize_adjacency(graph.adjacency, "symmetric" if method == "symmetric" else "row")
    
    def deriv(features):
        d, _, _ = grand_derivative(features, adj_norm)
        return d
    
    if integrator == "euler":
        times, features = euler_integrate(deriv, config.initial_features, config.dt, config.duration)
    else:
        times, features = rk4_integrate(deriv, config.initial_features, config.dt, config.duration)
    
    n_times = len(times)
    feat_derivs = np.zeros((n_times, n))
    msg_matrices = np.zeros((n_times, n, n))
    variances = np.zeros(n_times)
    energies = np.zeros(n_times)
    
    for t_idx in range(n_times):
        f = features[t_idx]
        d, m, _ = grand_derivative(f, adj_norm)
        feat_derivs[t_idx] = d
        msg_matrices[t_idx] = m
        variances[t_idx] = np.var(f)
        energies[t_idx] = dirichlet_energy(f, graph.adjacency)
    
    return GrandResult(times, features, feat_derivs, msg_matrices, variances, energies, adj_norm, config)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def save_plot(fig, filename, dpi=150):
    """Save plot with tight layout and close it."""
    fig.tight_layout()
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {filename}")

def plot_graph(graph, phases=None, features=None, filename=None):
    """Plot 3-node graph with edge weights."""
    fig, ax = plt.subplots(figsize=(10, 8))
    names = list(graph.node_names)
    adj = graph.adjacency
    pos = {names[0]: (0, 1), names[1]: (-0.866, -0.5), names[2]: (0.866, -0.5)}
    
    # Edges
    for i in range(3):
        for j in range(i+1, 3):
            w = adj[i, j]
            if w > 0:
                s, e = pos[names[i]], pos[names[j]]
                lw = 1 + 4*w
                ax.plot([s[0], e[0]], [s[1], e[1]], color="gray", linewidth=lw, alpha=0.7)
                m = ((s[0]+e[0])/2, (s[1]+e[1])/2)
                ax.text(m[0], m[1], f"{w:.1f}", ha="center", va="center", 
                       bbox=dict(facecolor="white", alpha=0.8, pad=1))
    
    # Nodes
    for i, name in enumerate(names):
        x, y = pos[name]
        if phases is not None:
            color = plt.cm.hsv((phases[i] % (2*np.pi)) / (2*np.pi))
            label = f"{name}\nθ={phases[i]:.2f}"
        elif features is not None:
            norm = (features[i] - min(features)) / (max(features) - min(features) + 1e-10)
            color = plt.cm.coolwarm(norm)
            label = f"{name}\nX={features[i]:.2f}"
        else:
            color = "lightgray"; label = name
        ax.scatter([x], [y], s=500, c=[color], edgecolor="black", linewidth=2)
        ax.text(x, y, label, ha="center", va="center", 
               bbox=dict(facecolor="white", alpha=0.8, pad=2))
    
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color="gray",lw=1,label="Weak (0.2)"),
                      Line2D([0],[0],color="gray",lw=2.6,label="Medium (0.6)"),
                      Line2D([0],[0],color="gray",lw=5,label="Strong (1.0)")],
           loc="upper right", title="Edge Weights")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Three-Node Graph\nwith Initial Values", fontsize=16, pad=20)
    if filename: save_plot(fig, filename)
    return fig

def plot_adjacency(graph, filename=None):
    """Plot adjacency matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    adj = graph.adjacency
    names = list(graph.node_names)
    im = ax.imshow(adj, cmap="YlOrBr", vmin=0, vmax=max(adj.flatten())*1.1)
    for i in range(3):
        for j in range(3):
            txt = f"{adj[i,j]:.1f}" if i != j else "0"
            color = "black" if adj[i,j] < 0.5 else "white"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=14)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(names); ax.set_yticklabels(names)
    ax.set_title("Adjacency Matrix\n(A[i,j] = weight from j to i)", fontsize=16)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Edge Weight")
    if filename: save_plot(fig, filename)
    return fig

def plot_phase_trajectories(result, graph, filename=None):
    """Plot unwrapped phase trajectories."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(graph.node_names):
        ax.plot(result.times, result.phases[:, i], label=f"Node {name}", linewidth=2)
    ax.set_xlabel("Time"); ax.set_ylabel("Phase (radians, unwrapped)")
    ax.set_title("Kuramoto Phase Trajectories\n(Unwrapped)", fontsize=16)
    ax.legend(); ax.grid(True, alpha=0.3)
    if filename: save_plot(fig, filename)
    return fig

def plot_wrapped_phases(result, graph, filename=None):
    """Plot wrapped phases [0, 2π)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    wrapped = result.phases % (2 * np.pi)
    for i, name in enumerate(graph.node_names):
        ax.plot(result.times, wrapped[:, i], label=f"Node {name}", linewidth=2)
    ax.set_xlabel("Time"); ax.set_ylabel("Phase (radians, [0, 2π))")
    ax.set_title("Kuramoto Wrapped Phases\n([0, 2π) convention)", fontsize=16)
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 2*np.pi)
    ax.axhline(y=2*np.pi, color="gray", linestyle="--", alpha=0.5)
    if filename: save_plot(fig, filename)
    return fig

def plot_unit_circle(result, graph, times=None, filename=None):
    """Plot oscillators on unit circle at snapshots."""
    if times is None:
        times = [0.0, result.times[len(result.times)//4], result.times[-1]]
    fig, axes = plt.subplots(1, len(times), figsize=(5*len(times), 5))
    if len(times) == 1: axes = [axes]
    colors = ["red", "green", "blue"]; names = list(graph.node_names)
    
    for ax_idx, t in enumerate(times):
        ax = axes[ax_idx]; t_idx = np.argmin(np.abs(result.times - t))
        phases = result.phases[t_idx]; r = result.order_parameters[t_idx]
        circle = Circle((0,0), 1, fill=False, color="gray", linewidth=2, alpha=0.5)
        ax.add_patch(circle)
        for i, (name, phase) in enumerate(zip(names, phases)):
            x, y = np.cos(phase), np.sin(phase)
            ax.quiver(0, 0, x, y, angles="xy", scale=1, scale_units="xy",
                     color=colors[i], width=0.015, label=f"{name} (θ={phase:.2f})")
            ax.scatter([x], [y], color=colors[i], s=100)
            ax.text(x*1.15, y*1.15, name, color=colors[i], fontsize=12)
        mx, my = np.mean(np.cos(phases)), np.mean(np.sin(phases))
        if mx**2 + my**2 > 1e-10:
            ax.quiver(0,0,mx,my,angles="xy",scale=1,scale_units="xy",
                     color="black",width=0.02,label=f"Mean (r={r:.3f})")
        ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
        ax.set_title(f"t={t:.2f}"); ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Kuramoto Oscillators on Unit Circle", fontsize=16, y=1.02)
    if filename: save_plot(fig, filename)
    return fig

def plot_order_parameter(results, graph, filename=None):
    """Plot order parameter for different couplings."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, result in results.items():
        ax.plot(result.times, result.order_parameters, label=label, linewidth=2)
    ax.set_xlabel("Time"); ax.set_ylabel("Order Parameter r(t)")
    ax.set_title("Kuramoto Synchronization\n(r≈0: dispersed, r≈1: synchronized)", fontsize=16)
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)
    if filename: save_plot(fig, filename)
    return fig


def plot_pairwise_diffs(result, graph, filename=None):
    """Plot wrapped phase differences between all pairs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    pairs = [("A-B", 0, 1), ("A-C", 0, 2), ("B-C", 1, 2)]
    for name, i, j in pairs:
        diff = result.phases[:, i] - result.phases[:, j]
        wrapped = np.array([wrapped_phase_difference(d, "pi") for d in diff])
        ax.plot(result.times, wrapped, label=name, linewidth=2)
    ax.set_xlabel("Time"); ax.set_ylabel("Wrapped Phase Difference (radians)")
    ax.set_title("Pairwise Wrapped Phase Differences\n([-π, π) convention)", fontsize=16)
    ax.legend(); ax.grid(True, alpha=0.3); ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(-np.pi, np.pi)
    if filename: save_plot(fig, filename)
    return fig

def plot_edge_messages(result, graph, filename=None):
    """Plot edge messages over time."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes_flat = axes.flatten()
    names = list(graph.node_names)
    edges = [("B→A", 0, 1), ("A→B", 1, 0), ("C→A", 0, 2), ("A→C", 2, 0), ("C→B", 1, 2), ("B→C", 2, 1)]
    
    for idx, (label, i, j) in enumerate(edges):
        ax = axes_flat[idx]
        msgs = result.message_matrices[:, i, j]
        ax.plot(result.times, msgs, color="blue", linewidth=2)
        ax.set_xlabel("Time"); ax.set_ylabel("Message")
        ax.set_title(f"{label}: {names[j]}→{names[i]}", fontsize=12)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Kuramoto Edge Messages\n(message from j to i: K·A[i,j]·sin(θ_j-θ_i))", fontsize=16, y=1.02)
    if filename: save_plot(fig, filename)
    return fig

def plot_message_snapshot(result, graph, t=None, filename=None):
    """Plot message matrix snapshot."""
    if t is None: t = result.times[len(result.times)//2]
    t_idx = np.argmin(np.abs(result.times - t))
    msgs = result.message_matrices[t_idx]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    names = list(graph.node_names)
    im = ax.imshow(msgs, cmap="RdBu", vmin=-abs(msgs).max(), vmax=abs(msgs).max())
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{msgs[i,j]:.3f}", ha="center", va="center", 
                   color="black" if abs(msgs[i,j]) < 0.5 else "white", fontsize=12)
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([f"from {n}" for n in names]); ax.set_yticklabels([f"to {n}" for n in names])
    ax.set_title(f"Message Matrix at t={t:.2f}\n(messages[i,j] = from j to i)", fontsize=14)
    fig.colorbar(im, ax=ax, shrink=0.8).set_label("Message Value")
    if filename: save_plot(fig, filename)
    return fig

def plot_frequency_coupling(result, graph, filename=None):
    """Plot freq vs coupling vs total derivative."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    names = list(graph.node_names)
    omega = result.config.natural_frequencies
    
    for i, name in enumerate(names):
        nat = omega[i] * np.ones_like(result.times)
        axes[0].plot(result.times, nat, label=name, linewidth=2)
        axes[1].plot(result.times, result.coupling_terms[:, i], label=name, linewidth=2)
        axes[2].plot(result.times, result.phase_velocities[:, i], label=name, linewidth=2)
    
    axes[0].set_ylabel("Natural Frequency\n(ω)"); axes[0].set_title("Natural Frequency Term")
    axes[1].set_ylabel("Coupling Term\n(Σ K·A·sin(Δθ))"); axes[1].set_title("Coupling Contribution") 
    axes[2].set_ylabel("Phase Velocity\n(dθ/dt = ω + coupling)"); axes[2].set_xlabel("Time"); axes[2].set_title("Total Derivative")
    for ax in axes: ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle("Kuramoto Equation Decomposition\ndθ_i/dt = ω_i + K Σ_j A_ij sin(θ_j-θ_i)", fontsize=16, y=1.02)
    if filename: save_plot(fig, filename)
    return fig

def plot_grand_trajectories(result, graph, filename=None):
    """Plot GRAND feature trajectories."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, name in enumerate(graph.node_names):
        ax.plot(result.times, result.features[:, i], label=f"Node {name}", linewidth=2)
    ax.set_xlabel("Time"); ax.set_ylabel("Feature Value")
    ax.set_title("GRAND Feature Trajectories\n(Linear Diffusion)", fontsize=16)
    ax.legend(); ax.grid(True, alpha=0.3)
    if filename: save_plot(fig, filename)
    return fig

def plot_grand_edge_messages(result, graph, filename=None):
    """Plot GRAND edge messages."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    axes_flat = axes.flatten()
    names = list(graph.node_names)
    edges = [("B→A", 0, 1), ("A→B", 1, 0), ("C→A", 0, 2), ("A→C", 2, 0), ("C→B", 1, 2), ("B→C", 2, 1)]
    
    for idx, (label, i, j) in enumerate(edges):
        ax = axes_flat[idx]
        msgs = result.message_matrices[:, i, j]
        ax.plot(result.times, msgs, color="green", linewidth=2)
        ax.set_xlabel("Time"); ax.set_ylabel("Message")
        ax.set_title(f"{label}: {names[j]}→{names[i]}", fontsize=12)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("GRAND Edge Messages\n(message from j to i: Â[i,j]·X_j)", fontsize=16, y=1.02)
    if filename: save_plot(fig, filename)
    return fig

def plot_grand_metrics(result, graph, filename=None):
    """Plot variance and Dirichlet energy."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(result.times, result.variances, color="purple", linewidth=2)
    axes[0].set_ylabel("Feature Variance"); axes[0].set_title("Feature Variance Across Nodes")
    axes[1].plot(result.times, result.dirichlet_energies, color="orange", linewidth=2)
    axes[1].set_ylabel("Dirichlet Energy"); axes[1].set_xlabel("Time"); axes[1].set_title("Graph Dirichlet Energy")
    for ax in axes: ax.grid(True, alpha=0.3)
    fig.suptitle("GRAND Smoothing Metrics\n(Diffusion reduces differences)", fontsize=16, y=1.02)
    if filename: save_plot(fig, filename)
    return fig

def plot_comparison(kur_result, grand_result, graph, filename=None):
    """Compare Kuramoto and GRAND."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    names = list(graph.node_names)
    
    for i, name in enumerate(names):
        axes[0,0].plot(kur_result.times, kur_result.phases[:, i], label=name, linewidth=2)
    axes[0,0].set_title("Kuramoto Phases"); axes[0,0].set_ylabel("Phase"); axes[0,0].legend()
    
    axes[0,1].plot(kur_result.times, kur_result.order_parameters, color="red", linewidth=2)
    axes[0,1].set_title("Kuramoto Sync (r)"); axes[0,1].set_ylabel("r"); axes[0,1].set_ylim(0, 1.05)
    
    for i, name in enumerate(names):
        axes[1,0].plot(grand_result.times, grand_result.features[:, i], label=name, linewidth=2)
    axes[1,0].set_title("GRAND Features"); axes[1,0].set_ylabel("Feature"); axes[1,0].set_xlabel("Time"); axes[1,0].legend()
    
    axes[1,1].plot(grand_result.times, grand_result.variances, color="purple", linewidth=2)
    axes[1,1].set_title("GRAND Variance"); axes[1,1].set_ylabel("Variance"); axes[1,1].set_xlabel("Time")
    
    for ax in axes.flatten(): ax.grid(True, alpha=0.3)
    fig.suptitle("Kuramoto vs GRAND\nPhase synchronization vs feature smoothing", fontsize=16, y=1.02)
    if filename: save_plot(fig, filename)
    return fig


# =============================================================================
# DATA EXPORT
# =============================================================================

def save_json(summary: dict, filename: Path):
    """Save JSON summary."""
    def convert(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, Path): return str(o)
        return str(o)
    with open(filename, "w") as f:
        json.dump(summary, f, indent=2, default=convert)
    logger.info(f"Saved: {filename}")

def save_csv(data: np.ndarray, columns: list, filename: Path):
    """Save CSV using pandas."""
    try:
        import pandas as pd
        pd.DataFrame(data, columns=columns).to_csv(filename, index=False)
        logger.info(f"Saved: {filename}")
    except ImportError:
        logger.warning("Pandas not available, skipping CSV export")


# =============================================================================
# CLI AND MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple KuramoroGNN-GRAND: Educational Graph Dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", type=str, default="Model Simple KuramoroGNN-GRAND/results",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--duration", type=float, default=10.0, help="Simulation duration")
    parser.add_argument("--dt", type=float, default=0.01, help="Time step")
    parser.add_argument("--integrator", choices=["euler", "rk4"], default="euler", help="Integration method")
    parser.add_argument("--coupling", type=float, default=2.0, help="Default coupling strength")
    parser.add_argument("--weak-coupling", type=float, default=0.5, help="Weak coupling")
    parser.add_argument("--strong-coupling", type=float, default=5.0, help="Strong coupling")
    parser.add_argument("--edge-ab", type=float, default=1.0, help="Edge weight A-B")
    parser.add_argument("--edge-bc", type=float, default=0.6, help="Edge weight B-C")
    parser.add_argument("--edge-ac", type=float, default=0.2, help="Edge weight A-C")
    parser.add_argument("--quick", action="store_true", help="Quick mode (short duration)")
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    
    if args.quick:
        args.duration = 2.0
        args.dt = 0.05
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create graph
    adj = np.array([[0.0, args.edge_ab, args.edge_ac],
                    [args.edge_ab, 0.0, args.edge_bc],
                    [args.edge_ac, args.edge_bc, 0.0]], dtype=float)
    graph = GraphConfig(adjacency=adj)
    validate_adjacency(graph.adjacency, graph.node_names)
    
    logger.info(f"Graph: {graph.node_names}")
    logger.info(f"Adjacency:\n{graph.adjacency}")
    
    # Run simulations
    logger.info("Running Kuramoto simulations...")
    kur_results = {}
    
    for label, K in [("K=0", 0.0), ("K=weak", args.weak_coupling), ("K=default", args.coupling), ("K=strong", args.strong_coupling)]:
        config = KuramotoConfig(
            initial_phases=np.array([0.0, 2.0, 4.0]),
            natural_frequencies=np.array([1.00, 1.12, 0.90]),
            coupling_strength=K,
            dt=args.dt, duration=args.duration
        )
        kur_results[label] = simulate_kuramoto(config, graph, args.integrator)
    
    logger.info("Running GRAND simulation...")
    grand_config = GrandConfig(
        initial_features=np.array([1.0, -0.5, 0.25]),
        dt=args.dt, duration=args.duration
    )
    grand_result = simulate_grand(grand_config, graph, args.integrator)
    
    logger.info("Generating visualizations...")
    plot_files = []
    
    # Generate all plots
    plots = [
        ("01_three_node_graph.png", lambda: plot_graph(graph, kur_results["K=default"].config.initial_phases, grand_config.initial_features, output_dir / "01_three_node_graph.png")),
        ("02_adjacency_matrix.png", lambda: plot_adjacency(graph, output_dir / "02_adjacency_matrix.png")),
        ("03_kuramoto_phase_trajectories.png", lambda: plot_phase_trajectories(kur_results["K=default"], graph, output_dir / "03_kuramoto_phase_trajectories.png")),
        ("04_kuramoto_wrapped_phases.png", lambda: plot_wrapped_phases(kur_results["K=default"], graph, output_dir / "04_kuramoto_wrapped_phases.png")),
        ("05_kuramoto_unit_circle_snapshots.png", lambda: plot_unit_circle(kur_results["K=default"], graph, [0, args.duration/2, args.duration], output_dir / "05_kuramoto_unit_circle_snapshots.png")),
        ("06_kuramoto_order_parameter.png", lambda: plot_order_parameter({k:v for k,v in kur_results.items() if k != "K=default"}, graph, output_dir / "06_kuramoto_order_parameter.png")),
        ("07_pairwise_phase_differences.png", lambda: plot_pairwise_diffs(kur_results["K=default"], graph, output_dir / "07_pairwise_phase_differences.png")),
        ("08_kuramoto_edge_messages.png", lambda: plot_edge_messages(kur_results["K=default"], graph, output_dir / "08_kuramoto_edge_messages.png")),
        ("09_kuramoto_message_matrix_snapshot.png", lambda: plot_message_snapshot(kur_results["K=default"], graph, None, output_dir / "09_kuramoto_message_matrix_snapshot.png")),
        ("10_frequency_and_coupling_terms.png", lambda: plot_frequency_coupling(kur_results["K=default"], graph, output_dir / "10_frequency_and_coupling_terms.png")),
        ("11_grand_feature_trajectories.png", lambda: plot_grand_trajectories(grand_result, graph, output_dir / "11_grand_feature_trajectories.png")),
        ("12_grand_edge_messages.png", lambda: plot_grand_edge_messages(grand_result, graph, output_dir / "12_grand_edge_messages.png")),
        ("13_grand_smoothing_metrics.png", lambda: plot_grand_metrics(grand_result, graph, output_dir / "13_grand_smoothing_metrics.png")),
        ("14_kuramoto_vs_grand.png", lambda: plot_comparison(kur_results["K=default"], grand_result, graph, output_dir / "14_kuramoto_vs_grand.png")),
    ]
    
    for name, plot_func in plots:
        try:
            plot_func()
            plot_files.append(name)
        except Exception as e:
            logger.error(f"Failed to generate {name}: {e}")
    
    # Save data
    summary = {
        "args": {a: getattr(args, a) for a in vars(args)},
        "graph": {"adjacency": graph.adjacency.tolist(), "node_names": list(graph.node_names)},
        "kuramoto_default": {
            "final_phases": kur_results["K=default"].phases[-1].tolist(),
            "final_order_parameter": float(kur_results["K=default"].order_parameters[-1]),
        },
        "grand": {
            "final_features": grand_result.features[-1].tolist(),
            "final_variance": float(grand_result.variances[-1]),
            "final_dirichlet_energy": float(grand_result.dirichlet_energies[-1]),
        },
        "plots": plot_files,
    }
    save_json(summary, output_dir / "simulation_summary.json")
    
    # Save CSVs if pandas available
    try:
        import pandas as pd
        names = list(graph.node_names)
        
        # Kuramoto trajectories
        kur_data = np.column_stack([kur_results["K=default"].times, kur_results["K=default"].phases])
        kur_cols = ["time"] + [f"phase_{n}" for n in names]
        save_csv(kur_data, kur_cols, output_dir / "kuramoto_trajectories.csv")
        
        # GRAND trajectories  
        grand_data = np.column_stack([grand_result.times, grand_result.features])
        grand_cols = ["time"] + [f"feature_{n}" for n in names]
        save_csv(grand_data, grand_cols, output_dir / "grand_trajectories.csv")
    except ImportError:
        pass
    
    # Print report
    print("\n" + "="*60)
    print("SIMPLE KURAMOROGNN-GRAND: SIMULATION COMPLETE")
    print("="*60)
    print(f"Graph: {list(graph.node_names)}")
    print(f"Adjacency:\n{graph.adjacency}")
    print(f"Coupling: {args.coupling} (default), {args.weak_coupling} (weak), {args.strong_coupling} (strong)")
    print(f"Time: dt={args.dt}, duration={args.duration}, integrator={args.integrator}")
    print("\nKuramoto Results:")
    for label, res in kur_results.items():
        print(f"  {label}: final r={res.order_parameters[-1]:.4f}")
    print("\nGRAND Results:")
    print(f"  Final features: {grand_result.features[-1]}")
    print(f"  Final variance: {grand_result.variances[-1]:.4f}")
    print(f"  Final energy: {grand_result.dirichlet_energies[-1]:.4f}")
    print(f"\nGenerated {len(plot_files)} plots")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()