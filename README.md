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
- `Model Simple KuramoroGNN-GRAND/` — educational demonstration of Kuramoto oscillators and GRAND diffusion on a three-node synthetic graph.
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

### Model Simple KuramoroGNN-GRAND

`Model Simple KuramoroGNN-GRAND/` is an **educational demonstration** that illustrates two fundamental graph-based dynamical systems on a simple three-node synthetic graph:

1. **Kuramoto oscillators**: Phase-coupled oscillators that synchronize through graph edges
2. **GRAND diffusion**: Linear graph diffusion that smooths node features

This model emphasizes **pedagogical clarity** over scientific realism. It uses a synthetic three-node graph with configurable edge weights to demonstrate:

- Graph construction and adjacency matrix validation
- Node states (phases vs features) and their mathematical meaning
- Message passing patterns in both systems
- Continuous-time graph dynamics via explicit Euler and RK4 integration
- Synchronization metrics (Kuramoto order parameter)
- Smoothing metrics (feature variance, Dirichlet energy)
- The connection between both systems and continuous-depth GNNs

**Key difference from Model KuramotoGNN**: This is purely synthetic and educational, while Model KuramotoGNN applies these concepts to real STEW EEG data for classification.

Run the educational demonstration from the repository root:

```bash
python "Model Simple KuramoroGNN-GRAND/simple_kuramoto_grand.py" --help
python "Model Simple KuramoroGNN-GRAND/simple_kuramoto_grand.py" --quick
```

Expected outputs include 14 educational plots (PNG files), JSON summary, and CSV trajectories in the specified output directory.

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

## Installation

Requires Python 3.10+. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"        # every pipeline + dev tools (pytest, ruff, mypy)
```

Or install only what one pipeline needs:

```bash
pip install -e ".[gat]"            # Model GAT
pip install -e ".[kuramotognn]"    # Model KuramotoGNN
pip install -e ".[subject-ident]"  # Model Subject Ident (also installs tensorflow/keras)
pip install -e ".[dev]"            # pytest, ruff, mypy
```

`eeg_config.py` at the repository root centralizes constants shared by all three
pipelines (sampling frequency, channel names/regions, frequency bands, window and
step duration, and `set_seed()` for Python/NumPy/PyTorch). Each pipeline script adds
the repo root to `sys.path` at import time so `from eeg_config import ...` works
whether the script is run directly or imported.

## Grouped evaluation methodology (leakage prevention)

STEW recordings are segmented into windows with 50% overlap (step < window), so a
random window-level train/test split can put near-duplicate, overlapping windows from
the same recording on both sides of the split — inflating reported accuracy. Every
window keeps its subject id, condition, recording id, window index, and start time.

- **Model KuramotoGNN** and **Model GAT** classify rest vs. high workload, so they use
  `sklearn.model_selection.GroupKFold` keyed by subject id: all of a subject's windows
  fall in exactly one fold, and fold-level metrics are aggregated as mean ± std.
- **Model Subject Ident** classifies *subject identity*, so leave-subject-out CV is not
  applicable (the subject is the label). Instead, each subject's windows are sorted in
  time and split into contiguous temporal blocks assigned wholesale to train,
  validation, and test, so overlapping/adjacent windows never cross a partition
  boundary.
- Feature scaling (`StandardScaler`) is fit on the training partition only in every
  pipeline.
- Reported metrics are window-level unless a README explicitly says recording- or
  subject-level (e.g. Model GAT's `recording_accuracy`/`recording_auc`, computed from
  the median predicted probability per recording).

**Limitation:** window-level accuracy — even under grouped CV — measures how well a
model distinguishes conditions/subjects *within this STEW cohort's recording
sessions*. It is not evidence of generalization to a new recording session, a new day,
or a different EEG headset/population.

## Testing, linting, and type checking

```bash
pip install -e ".[dev]"
python -m pytest -m "not slow"   # unit tests (seconds)
python -m pytest -m slow         # reduced synthetic end-to-end smoke tests per pipeline
python -m pytest                 # everything
ruff check .
ruff format --check .
mypy eeg_config.py "Model GAT/stew_asi_gat_experiment.py" "Model KuramotoGNN/kuramoto_gnn_stew.py" "Model Subject Ident/stewSubjectsIdentification.py"
```

Tests live in `tests/` and use small synthetic arrays and `tmp_path` — they do not
require the full STEW dataset. Slow tests run a reduced end-to-end pass of each
pipeline against a synthetic 2-3 subject dataset in a temporary directory and never
touch the checked-in `dataset/` or `Model */results` output.

## Notes on reproducibility

- Every pipeline accepts `--seed`, and `eeg_config.set_seed()` seeds Python's `random`,
  NumPy, and PyTorch; the subject-identification pipeline additionally seeds Keras
  training runs through the same call.
- Every pipeline writes a `summary.json` alongside its other output, recording the
  effective CLI arguments, seed, dataset file/subject/window counts, installed
  dependency versions, and evaluation metrics — so a result can be traced back to the
  exact configuration that produced it.
- For reproducible STEW analysis, keep the `dataset/` directory in place and use the
  folder-specific scripts as documented above; `Model KuramotoGNN/requirements.txt` is
  kept for backward compatibility but `pyproject.toml` is the source of truth for
  dependencies.

## References

- Lim, Sourina, and Wang, "STEW: Simultaneous Task EEG Workload Data Set," IEEE TNSRE, 2018.
- Kuramoto, Y. "Chemical Oscillations, Waves, and Turbulence," Springer, 1984.
- Kipf, T. N., and Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks," ICLR 2017.

