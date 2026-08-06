# STEW EEG Subject Identification and Graph Dynamics Modeling

This repository explores three complementary modeling approaches for the STEW EEG workload dataset:

1. `Model GAT/` — a latent attention graph-attention model that computes a rest-normalized ASI-EEG index using sensor-space EEG features.
2. `Model KuramotoGNN/` — a Kuramoto-inspired graph neural dynamics experiment comparing linear diffusion and phase-coupled oscillator propagation on STEW EEG windows.
3. `Model Subject Ident/` — a classical feature-extraction pipeline for subject identification using statistical, entropy, and fractal EEG features.

## Dataset

The STEW dataset is the Simultaneous Task EEG Workload Dataset published in 2018:

- W. L. Lim, O. Sourina and L. P. Wang, "STEW: Simultaneous Task EEG Workload Data Set," IEEE Transactions on Neural Systems and Rehabilitation Engineering, vol. 26, no. 11, pp. 2106-2114, Nov. 2018.
- DOI: 10.1109/TNSRE.2018.2872924
- Open access: https://ieee-dataport.org/open-access/stew-simultaneous-task-eeg-workload-dataset

### Dataset structure

The repository contains raw STEW recordings in `dataset/`.
Each file follows the naming convention:

- `subXX_lo.txt` — rest condition for subject XX
- `subXX_hi.txt` — high workload condition for subject XX

Each file is a time series matrix with 14 EEG channels in the following order:

`AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4`

Sampling frequency: 128 Hz.

## Repository structure

- `dataset/` — raw STEW EEG files + rating metadata.
- `Model GAT/` — latent graph-attention experiment using sensor-space EEG features and ASI-EEG.
- `Model KuramotoGNN/` — diffusion and phase-coupled oscillator modeling, including STEW-based dynamics evaluation.
- `Model Subject Ident/` — subject identification pipeline built from statistical and entropy-derived EEG features.

## Model descriptions

### Model GAT

`Model GAT/` implements a sensor-space graph-attention experiment that uses the 14 EEG channels as cortical nodes and adds a latent thalamic attention node.

Key contributions:

- window-level spectral and phase feature extraction from STEW EEG recordings
- rest normalization by subject to isolate workload-related changes
- computation of an ASI-EEG index for rest versus high workload comparison
- latent THAL attention modeling via a graph attention readout
- paired statistical evaluation and cross-validated classification of rest/high conditions

Common outputs:

- `Model GAT/results/summary.json`
- `Model GAT/results/ctx_node_mapping.csv`
- `Model GAT/results/subject_condition_metrics.csv`

Run the experiment from the repository root:

```bash
python "Model GAT/stew_asi_gat_experiment.py" --dataset dataset --output "Model GAT/results" --bootstrap 200
```

### Model KuramotoGNN

`Model KuramotoGNN/` examines graph-based neural dynamics in two complementary ways:

1. a synthetic Kuramoto-inspired GNN demo that generates node features, constructs a graph, and evaluates classification performance
2. a STEW-specific real-data pipeline that encodes EEG band-power windows, propagates node dynamics through a normalized channel graph, and compares two diffusion mechanisms:
   - linear GRAND diffusion: `dX/dt = (A_hat - I) X`
   - Kuramoto phase coupling: `dX/dt = omega + K * A_hat sin(X_j - X_i)`

This module emphasizes interpretable graph dynamics with metrics such as:

- mean pairwise distance
- Dirichlet energy
- class separation
- cosine similarity
- velocity synchronization

The STEW experiment also evaluates downstream classification performance using pooled node representations, and saves confusion matrix plots and connectivity visualizations.

Run the STEW dynamics experiment from the repository root:

```bash
python "Model KuramotoGNN/kuramoto_gnn_stew.py" --dataset dataset --output_dir "Model KuramotoGNN"
```

Expected outputs include:

- metric curves: `mean_pairwise_distance.png`, `dirichlet_energy.png`, `class_separation.png`, `cosine_similarity.png`
- representation analysis: `pca_features.png`
- graph visualizations: `networkx_graph_visulisation.png`, `node_connectivity_heatmap.png`
- classification heatmaps in `Model KuramotoGNN/test_output/`

### Model Subject Ident

`Model Subject Ident/` implements a classical EEG subject identification pipeline from STEW data.

Key elements:

- bandpass filtering and artifact-aware preprocessing of 14-channel EEG recordings
- extraction of per-channel statistical features (mean, variance, skewness, kurtosis)
- spectral power feature extraction using Welch PSD
- entropy and fractal dimension features (permutation entropy, spectral entropy, SVD entropy, approximate entropy, sample entropy, Petrosian/Katz/Higuchi fractal dimensions, detrended fluctuation analysis)
- feature aggregation and classification using traditional machine learning and neural-network methods

This folder also includes visualizations of model performance and confusion matrices.

Run the subject identification pipeline from the repository root:

```bash
python "Model Subject Ident/stewSubjectsIdentification.py"
```

## Notes on reproducibility

- The repository uses Python 3.x and standard scientific libraries such as `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`, `scikit-learn`, `keras`, and `mne`.
- `Model KuramotoGNN/requirements.txt` lists the dependencies for the Kuramoto dynamics module.
- For reproducible STEW analysis, keep the `dataset/` directory in place and use the folder-specific scripts as documented above.

## References

- Lim, Sourina, and Wang, "STEW: Simultaneous Task EEG Workload Data Set," IEEE TNSRE, 2018.
- Kuramoto, Y. "Chemical Oscillations, Waves, and Turbulence," Springer, 1984.
- Kipf, T. N., and Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks," ICLR 2017.

