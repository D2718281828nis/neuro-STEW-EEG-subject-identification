# Simple KuramoroGNN-GRAND: Educational Graph Dynamics Demonstration

**Simple KuramoroGNN-GRAND** is an educational Python module that demonstrates and visualizes two fundamental graph-based dynamical systems on a simple three-node graph:

1. **Kuramoto oscillators**: Phase-coupled oscillators that can synchronize through graph edges
2. **GRAND (Graph Random Neural Dynamics)**: Linear graph diffusion that smooths node features

Both systems are implemented as continuous-time dynamical systems and illustrate how graph message passing works in different contexts.

---

## Purpose and Educational Goals

This module is designed to help understand:

- **Graph construction**: How to build and validate undirected weighted graphs
- **Node states and edge weights**: What these represent in different dynamical systems
- **Message passing**: How neighboring nodes exchange information through edges
- **Graph evolution**: How systems evolve over time on graphs
- **Kuramoto synchronization**: How coupled oscillators can synchronize their phases
- **GRAND diffusion**: How linear diffusion smooths features across graphs
- **System comparison**: Key differences between phase coupling and feature diffusion
- **GNN connection**: How both relate to continuous-depth graph neural networks

> **Important**: This is a **synthetic educational example**. It does not use the STEW EEG dataset, is not a trained classifier, and is not intended for scientific research without proper validation.

---

## Quick Start

```bash
# From the repository root:
python "Model Simple KuramoroGNN-GRAND/simple_kuramoto_grand.py" --help

# Run a quick demonstration (2 seconds, fast):
python "Model Simple KuramoroGNN-GRAND/simple_kuramoto_grand.py" --quick

# Run with custom parameters:
python "Model Simple KuramoroGNN-GRAND/simple_kuramoto_grand.py" \
    --duration 5 \
    --dt 0.01 \
    --coupling 3.0 \
    --output-dir ./my_results
```

---

## The Three-Node Graph

The example uses a simple undirected graph with three nodes labeled **A**, **B**, and **C**.

### Graph Structure

```
        A
       / \
      /   \
 1.0/     \0.2
    /       \
   B---------C
     0.6
```

### Adjacency Matrix

The graph is defined by this **3×3 adjacency matrix**:

|       | from A | from B | from C |
|-------|--------|--------|--------|
| **to A**   |   0.0  |   1.0  |   0.2  |
| **to B**   |   1.0  |   0.0  |   0.6  |
| **to C**   |   0.2  |   0.6  |   0.0  |

**Interpretation**:
- **A[i,j] = weight of edge from node j to node i**
- **Diagonal entries are zero**: No self-loops (A[i,i] = 0)
- **Matrix is symmetric**: For undirected graphs, A[i,j] = A[j,i]
- **Edge weights**: A-B = 1.0 (strong), B-C = 0.6 (medium), A-C = 0.2 (weak)

### What the Adjacency Matrix Means

Each entry A[i,j] represents the strength of the connection **from node j to node i**:

- **A[0,1] = 1.0**: Strong connection from B to A
- **A[1,2] = 0.6**: Medium connection from C to B  
- **A[0,2] = 0.2**: Weak connection from C to A
- **A[1,0] = 1.0**: Strong connection from A to B (same as A[0,1] due to symmetry)

### Degree Matrix

The degree of each node is the sum of its connections:

- **deg(A) = A[0,1] + A[0,2] = 1.0 + 0.2 = 1.2**
- **deg(B) = A[1,0] + A[1,2] = 1.0 + 0.6 = 1.6**
- **deg(C) = A[2,0] + A[2,1] = 0.2 + 0.6 = 0.8**

The degree matrix is:
```
D = [[1.2, 0.0, 0.0],
     [0.0, 1.6, 0.0],
     [0.0, 0.0, 0.8]]
```

---

## Initial Node Data

### For Kuramoto Oscillators

Each node has:

| Node | Initial Phase θ | Natural Frequency ω |
|------|-----------------|---------------------|
| A    | 0.0 rad         | 1.00 rad/time       |
| B    | 2.0 rad         | 1.12 rad/time       |
| C    | 4.0 rad         | 0.90 rad/time       |

**Units**:
- **Phase (θ)**: radians, periodic with period 2π
- **Angular frequency (ω)**: radians per unit time
- **Time**: arbitrary units

### For GRAND Diffusion

Each node has an initial scalar feature:

| Node | Initial Feature X |
|------|------------------|
| A    | 1.0              |
| B    | -0.5            |
| C    | 0.25            |

**Units**: arbitrary (normalized for visualization)

---

## Kuramoto Oscillator Model

### The Kuramoto Equation

The **graph-coupled Kuramoto equation** governs how each oscillator's phase evolves:

```
dθ_i/dt = ω_i + K · Σ_j A[i,j] · sin(θ_j - θ_i)
```

**Where:**
- **dθ_i/dt**: Rate of change of phase for node i (phase velocity)
- **ω_i**: Natural angular frequency of node i
- **K**: Global coupling strength (scalar)
- **A[i,j]**: Weight of edge from node j to node i
- **sin(θ_j - θ_i)**: Sine of phase difference between nodes j and i

### Message Passing Interpretation

The Kuramoto model can be understood as a **continuous-time graph neural network**:

1. **Message**: `m_j→i = K · A[i,j] · sin(θ_j - θ_i)`
   - Node j sends a message to node i
   - Message strength depends on coupling K, edge weight A[i,j], and phase difference

2. **Aggregate**: `Σ_j m_j→i`
   - Node i receives messages from all neighbors
   - Sum over all j (all incoming edges)

3. **Update**: `dθ_i/dt = ω_i + Σ_j m_j→i`
   - Natural frequency drives the oscillator
   - Coupling messages pull phases toward synchronization

### Why Sine Coupling?

The sine function has crucial properties:
- **sin(0) = 0**: When phases are equal, no coupling force
- **sin(Δθ) ≈ Δθ for small Δθ**: Linear attraction for small differences  
- **sin(Δθ) ∈ [-1, 1]**: Bounded coupling strength
- **sin(-Δθ) = -sin(Δθ)**: Direction matters (push vs pull)

### Synchronization and the Order Parameter

The **Kuramoto order parameter** `r(t)` measures synchronization:

```
r(t) = |(1/N) · Σ_j exp(i · θ_j(t))|
```

**Interpretation:**
- **r ≈ 0**: Phases are dispersed (no synchronization)
- **r ≈ 1**: Phases are synchronized (all oscillators in step)

**Geometric meaning**: `r` is the length of the mean phase vector on the unit circle.

### Hand-Worked Kuramoto Update Example

Let's compute one time step manually using the default values at t=0:

**Given:**
- Phases: θ_A = 0.0, θ_B = 2.0, θ_C = 4.0
- Natural frequencies: ω_A = 1.00, ω_B = 1.12, ω_C = 0.90
- Coupling: K = 2.0
- Adjacency: A[0,1] = 1.0 (A-B), A[0,2] = 0.2 (A-C), A[1,2] = 0.6 (B-C)

**Messages to Node A (i=0):**

- **From B to A**: m_B→A = K · A[0,1] · sin(θ_B - θ_A) = 2.0 · 1.0 · sin(2.0 - 0.0) = 2.0 · sin(2.0) ≈ 2.0 · 0.9093 ≈ 1.8186
- **From C to A**: m_C→A = K · A[0,2] · sin(θ_C - θ_A) = 2.0 · 0.2 · sin(4.0 - 0.0) = 0.4 · sin(4.0) ≈ 0.4 · (-0.7568) ≈ -0.3027

**Total coupling to A**: Σ messages = 1.8186 + (-0.3027) ≈ 1.5159

**Phase derivative for A**: dθ_A/dt = ω_A + coupling = 1.00 + 1.5159 ≈ **2.5159 rad/time**

Similarly, we can compute for B and C:

- **Messages to B**: From A = 2.0·1.0·sin(0.0-2.0) ≈ -1.8186, From C = 2.0·0.6·sin(4.0-2.0) ≈ 2.0·0.6·0.9093 ≈ 1.0912
- **Coupling to B**: -1.8186 + 1.0912 ≈ -0.7274
- **dθ_B/dt**: 1.12 + (-0.7274) ≈ **0.3926 rad/time**

- **Messages to C**: From A = 2.0·0.2·sin(0.0-4.0) ≈ 2.0·0.2·(-0.7568) ≈ 0.3027, From B = 2.0·0.6·sin(2.0-4.0) ≈ 2.0·0.6·(-0.9093) ≈ -1.0912
- **Coupling to C**: 0.3027 + (-1.0912) ≈ -0.7885
- **dθ_C/dt**: 0.90 + (-0.7885) ≈ **0.1115 rad/time**

### Coupling Scenarios

The script runs four coupling scenarios:

| Coupling | K value | Behavior |
|----------|---------|----------|
| **K=0** | 0.0 | Uncoupled: oscillators drift apart at their natural frequencies |
| **K=weak** | 0.5 | Weak synchronization: partial phase alignment |
| **K=default** | 2.0 | Strong synchronization: phases converge rapidly |
| **K=strong** | 5.0 | Very strong: almost immediate synchronization |

---

## GRAND Diffusion Model

### The GRAND Equation

GRAND (Graph Random Neural Dynamics) implements **linear graph diffusion**:

```
dX/dt = (Â - I) · X = -L · X
```

**Where:**
- **X**: Feature vector [X_A, X_B, X_C]^T
- **Â**: Normalized adjacency matrix (we use symmetric normalization)
- **I**: Identity matrix
- **L**: Graph Laplacian matrix

### Normalized Adjacency

We use **symmetric normalization**:

```
Â = D^(-1/2) · A · D^(-1/2)
```

**Where:**
- **D**: Degree matrix (diagonal)
- **D^(-1/2)**: Diagonal matrix with 1/√deg(i) on diagonal

For our default graph:
- deg(A) = 1.2, deg(B) = 1.6, deg(C) = 0.8
- D^(-1/2) = diag([1/√1.2, 1/√1.6, 1/√0.8]) ≈ diag([0.9129, 0.7906, 1.1180])

### Message Passing Interpretation

GRAND can also be understood as message passing:

1. **Message**: `m_j→i = Â[i,j] · X_j`
   - Node j sends its feature value scaled by edge weight

2. **Aggregate**: `Σ_j m_j→i = Σ_j Â[i,j] · X_j`
   - Node i receives weighted contributions from all neighbors

3. **Update**: `dX_i/dt = (Σ_j Â[i,j] · X_j) - X_i`
   - Feature moves toward the weighted average of neighbors
   - Minus X_i: subtracts current value (like a gradient step)

### Hand-Worked GRAND Update Example

Let's compute the normalized adjacency and one update:

**Given:**
- Initial features: X = [1.0, -0.5, 0.25]
- Adjacency: A = [[0, 1.0, 0.2], [1.0, 0, 0.6], [0.2, 0.6, 0]]

**Normalized adjacency (symmetric):**

First compute degrees: deg_A = 1.2, deg_B = 1.6, deg_C = 0.8

D^(-1/2) = diag([1/√1.2, 1/√1.6, 1/√0.8])

Â = D^(-1/2) · A · D^(-1/2):

```
Â ≈ [[0.000, 0.693, 0.192],
     [0.693, 0.000, 0.425],
     [0.192, 0.425, 0.000]]
```

**Messages to Node A (i=0):**

- **From B to A**: m_B→A = Â[0,1] · X_B = 0.693 · (-0.5) ≈ -0.3465
- **From C to A**: m_C→A = Â[0,2] · X_C = 0.192 · 0.25 ≈ 0.0480

**Aggregated message to A**: Σ messages = -0.3465 + 0.0480 ≈ -0.2985

**Feature derivative for A**: dX_A/dt = aggregated - X_A = -0.2985 - 1.0 = **-1.2985**

This means node A's feature will decrease rapidly (moving toward the average).

### Smoothing Properties

GRAND diffusion has important properties:

- **Preserves constant features**: If all X_i are equal, dX/dt = 0
- **Reduces variance**: Feature differences decrease over time
- **Minimizes Dirichlet energy**: E = Σ_{i,j} A[i,j] (X_i - X_j)²

The **Dirichlet energy** measures the total feature variation across edges. As diffusion progresses, this energy decreases, indicating smoother features.

---

## Key Differences: Kuramoto vs GRAND

| Aspect | Kuramoto | GRAND |
|--------|----------|-------|
| **State variable** | Phase θ (periodic, radians) | Feature X (scalar, arbitrary) |
| **Dynamics** | Nonlinear (sine coupling) | Linear (diffusion) |
| **Message function** | K·A·sin(θ_j-θ_i) | Â·X_j |
| **Update rule** | dθ/dt = ω + coupling | dX/dt = (Â-I)X |
| **Fixed points** | Synchronized phases | Equal features |
| **Synchronization metric** | Order parameter r | Feature variance |
| **Energy metric** | N/A | Dirichlet energy |
| **Interpretation** | Phase oscillators synchronize | Features smooth/average |

### GNN Connection

Both systems demonstrate **continuous-time graph message passing**:

```
# Generic GNN message passing pattern:
for each node i:
    messages = [f(node_j, node_i, edge_ij) for all neighbors j]
    aggregated = sum(messages)
    update = g(node_i, aggregated)
    node_i = node_i + dt * update
```

- **Kuramoto**: f(j,i) = K·A_ij·sin(θ_j-θ_i), g(i,agg) = ω_i + agg
- **GRAND**: f(j,i) = Â_ij·X_j, g(i,agg) = agg - X_i

---

## Visualizations

The script generates **14 educational plots** in the output directory:

### Graph Structure
- **01_three_node_graph.png**: The 3-node graph with edge weights and initial values
- **02_adjacency_matrix.png**: Heatmap of the adjacency matrix

### Kuramoto Oscillators
- **03_kuramoto_phase_trajectories.png**: Unwrapped phase vs time for all nodes
- **04_kuramoto_wrapped_phases.png**: Phases wrapped to [0, 2π) vs time
- **05_kuramoto_unit_circle_snapshots.png**: Oscillators on unit circle at multiple time points
- **06_kuramoto_order_parameter.png**: Synchronization r(t) for different coupling strengths
- **07_pairwise_phase_differences.png**: Wrapped phase differences between all pairs
- **08_kuramoto_edge_messages.png**: Message strength over time for all directed edges
- **09_kuramoto_message_matrix_snapshot.png**: Message matrix heatmap at one time point
- **10_frequency_and_coupling_terms.png**: Decomposition of Kuramoto equation terms

### GRAND Diffusion
- **11_grand_feature_trajectories.png**: Feature values vs time for all nodes
- **12_grand_edge_messages.png**: Edge messages over time for GRAND
- **13_grand_smoothing_metrics.png**: Feature variance and Dirichlet energy over time

### Comparison
- **14_kuramoto_vs_grand.png**: Multi-panel comparison of both systems

---

## Output Files

In addition to the 14 plots, the script generates:

- **simulation_summary.json**: Machine-readable summary with all parameters and results
- **kuramoto_trajectories.csv**: Phase trajectories as CSV
- **grand_trajectories.csv**: Feature trajectories as CSV

---

## CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--output-dir` | str | `Model Simple KuramoroGNN-GRAND/results` | Output directory for plots and data |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--duration` | float | 10.0 | Total simulation time |
| `--dt` | float | 0.01 | Time step for integration |
| `--integrator` | str | `euler` | Integration method: `euler` or `rk4` |
| `--coupling` | float | 2.0 | Default coupling strength K |
| `--weak-coupling` | float | 0.5 | Weak coupling for comparison |
| `--strong-coupling` | float | 5.0 | Strong coupling for comparison |
| `--edge-ab` | float | 1.0 | Edge weight A-B |
| `--edge-bc` | float | 0.6 | Edge weight B-C |
| `--edge-ac` | float | 0.2 | Edge weight A-C |
| `--quick` | flag | False | Quick mode: short duration, larger dt |

---

## Dependencies

The minimal dependencies are:

```
numpy>=1.26
matplotlib>=3.7
```

For CSV export (optional):
```
pandas>=2.0
```

For animation (future feature):
```
Pillow>=10.0
```

Install all dependencies:
```bash
pip install numpy matplotlib pandas Pillow
```

---

## Numerical and Scientific Limitations

This is an **educational demonstration** with several important limitations:

### Numerical Limitations
- **Explicit Euler integration**: Simple but can be unstable for large time steps
- **No adaptive stepping**: Fixed time step may miss rapid transients
- **Finite precision**: Floating-point arithmetic introduces small errors
- **Memory usage**: Stores full trajectories (not streaming for long runs)

### Scientific Limitations
- **Synthetic data**: All values are chosen for demonstration, not scientific accuracy
- **Small graph**: Only 3 nodes - real systems have hundreds or thousands
- **Simple dynamics**: Real Kuramoto systems may have additional terms
- **No noise**: Deterministic - real systems often have stochastic elements
- **No validation**: Not compared against real-world data

### Educational Focus
Despite these limitations, the model successfully demonstrates:
- ✅ Graph construction and validation
- ✅ Message passing on graphs
- ✅ Continuous-time dynamics
- ✅ Phase synchronization vs feature smoothing
- ✅ Numerical integration basics
- ✅ Connection to GNN concepts

---

## Repository Integration

This educational model is self-contained and does not modify or depend on:
- The STEW EEG dataset
- Existing model logic
- External data sources

It can be run independently, studied, modified, or removed without affecting other parts of the repository.

---

## References

- Kuramoto, Y. "Chemical Oscillations, Waves, and Turbulence" Springer, 1984
- Kipf, T. N., and Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks" ICLR 2017
- Hamilton, W. L. et al. "Inductive Representation Learning on Large Graphs" ICLR 2017