# Model KuramotoGNN

This folder contains two complementary Kuramoto-inspired graph modeling experiments:

- `kuramoto_gnn_example.py` — a synthetic demonstrator for Kuramoto-like node features, graph propagation, and binary classification.
- `kuramoto_gnn_stew.py` — a STEW EEG experiment that compares linear diffusion and Kuramoto phase coupling on real EEG channel graphs.

## What this folder studies

The core hypothesis is that EEG sensor-space connectivity can be modeled as graph dynamics, and that phase-coupled oscillator propagation may capture structure that is not present in linear diffusion alone.

The STEW experiment performs:

- band-power-based encoding of 14-channel EEG windows
- normalization of features within each subject
- construction of a normalized 14-channel adjacency matrix
- integration of node dynamics under two mechanisms:
  - GRAND diffusion: `dX/dt = (A_hat - I) X`
  - Kuramoto coupling: `dX/dt = omega + K * A_hat sin(X_j - X_i)`
- pooling of node representations for binary rest/high classification
- evaluation with confusion matrices, PCA projections, and graph-level metrics
## Why `KuramotoGNN identical omega` is weak

The `KuramotoGNN identical omega` variant sets the natural frequency `omega` to the same value for every node. In Kuramoto dynamics, identical natural frequencies remove one key source of heterogeneity across nodes, which causes the oscillators to synchronize too quickly and homogeneously. This oversynchronization collapses class-specific structure in the pooled representation and reduces the downstream classifier's ability to separate rest versus high workload states.

By contrast, the `KuramotoGNN nonidentical omega` version preserves node-wise frequency variation, which maintains richer phase dynamics and helps retain discriminative information for classification.
## Files

- `kuramoto_gnn_example.py` — synthetic Kuramoto-GNN demo script.
- `kuramoto_gnn_stew.py` — STEW EEG dynamics and classification experiment.
- `requirements.txt` — dependencies for this folder.
- `node_connectivity_heatmap.png` — adjacency heatmap visualization produced by the STEW pipeline.
- `test_output/` — saved confusion matrix figures and additional output.

## Run

From the repository root:

```bash
python "Model KuramotoGNN/kuramoto_gnn_stew.py" --dataset dataset --output_dir "Model KuramotoGNN"
```

## Expected outputs

The STEW experiment writes metric curves and visualizations such as:

- `mean_pairwise_distance.png`
- `dirichlet_energy.png`
- `class_separation.png`
- `cosine_similarity.png`
- `pca_features.png`
- `networkx_graph_visulisation.png`
- `node_connectivity_heatmap.png`
- `test_output/confusion_matrix_*.png`

