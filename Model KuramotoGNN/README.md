# Model KuramotoGNN

This folder contains two complementary Kuramoto-inspired graph modeling experiments:

- `kuramoto_gnn_example.py` — a synthetic demonstrator for Kuramoto-like node features, graph propagation, and binary classification.
- `kuramoto_gnn_stew.py` — a STEW EEG experiment that compares linear diffusion and Kuramoto phase coupling on real EEG channel graphs.

## What this folder studies

The core hypothesis is that EEG sensor-space connectivity can be modeled as graph dynamics, and that phase-coupled oscillator propagation may capture structure that is not present in linear diffusion alone.

The STEW experiment performs:

- band-power-based encoding of 14-channel EEG windows (window/step duration shared
  with the other two pipelines via the root `eeg_config.py`)
- normalization of features within each subject, relative to that subject's own rest baseline
- construction of a normalized 14-channel adjacency matrix
- integration of node dynamics under two mechanisms:
  - GRAND diffusion: `dX/dt = (A_hat - I) X`
  - Kuramoto coupling: `dX/dt = omega + K * A_hat sin(X_j - X_i)`
- pooling of node representations for binary rest/high classification
- evaluation with a **subject-grouped** cross-validated logistic classifier
  (`sklearn.model_selection.GroupKFold`), confusion matrices, PCA projections, and
  graph-level metrics

### Evaluation methodology and leakage prevention

Windows are overlapping (50% step), so a naive random train/test split lets
near-duplicate windows from the same subject appear on both sides — inflating
accuracy. Every window keeps its subject id, condition, recording id, window index,
and start time (written to `window_metadata.csv`); classification uses
`GroupKFold` keyed by subject id, so **all of a subject's windows fall in exactly
one fold**. Metrics are averaged across folds (mean ± std) and also reported as a
single out-of-fold aggregate; AUC is computed with `sklearn.metrics.roc_auc_score`
and is `NaN` for any fold whose held-out windows are all one class. All reported
metrics are **window-level** — this script does not pool predictions to the
recording or subject level.
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

Useful CLI options (see `--help` for the full list): `--sample_windows`, `--feature_dim`,
`--K`, `--dt`, `--max_T`, `--seed`, `--cv_splits` (subject-grouped CV fold count).

## Expected outputs

The STEW experiment writes metric curves, visualizations, and metadata such as:

- `mean_pairwise_distance.png`
- `dirichlet_energy.png`
- `class_separation.png`
- `cosine_similarity.png`
- `pca_features.png`
- `networkx_graph_visulisation.png`
- `node_connectivity_heatmap.png`
- `test_output/confusion_matrix_*.png` — out-of-fold confusion matrix at `T=max_T`
- `window_metadata.csv` — subject/condition/recording/window-index/start-time for every window used
- `summary.json` — effective CLI configuration, dataset/subject/window counts, dependency versions, and window-level fold metrics for every (model, T) combination

