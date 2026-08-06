# STEW ASI + latent THAL attention model

This folder contains a cross-validated rest-vs-high-workload experiment that runs on
the local STEW dataset in this repository. Window/step duration and channel/band
definitions are shared with the other two pipelines via the root `eeg_config.py`.

## What it does

- reads the 14-channel STEW EEG recordings under `dataset/`
- extracts per-window band-power and phase features
- computes a subject-level rest-normalized ASI-EEG index
- fits a latent THAL graph-attention readout over the 14 measured CTX nodes
- evaluates rest vs high workload with paired statistics and subject-grouped
  cross-validated classification (`sklearn.model_selection.GroupKFold`, keyed by
  subject id, so a subject's windows never split across train and test)

## Run

From the repository root:

```bash
python "Model GAT/stew_asi_gat_experiment.py" --dataset dataset --output "Model GAT/results" --bootstrap 200
```

Useful CLI options (see `--help` for the full list): `--seed`, `--splits` (subject-grouped
CV folds, auto-reduced with a warning if fewer subjects are available), `--bootstrap`.

The run produces files such as:

- `Model GAT/results/summary.json`
- `Model GAT/results/ctx_node_mapping.csv`
- `Model GAT/results/subject_condition_metrics.csv`
- `Model GAT/results/window_asi.csv`
- `Model GAT/results/thal_cross_validated_windows.csv`

## Evaluation methodology

The `gat_condition_classifier` metrics in `summary.json` come from `GroupKFold`
cross-validation over windows, grouped by subject, reported at both window level
(`window_accuracy`, `window_auc`) and recording level (`recording_accuracy`,
`recording_auc`, computed from the median predicted probability per recording). The
paired rest-vs-high statistics (Wilcoxon signed-rank test, bootstrap CIs, rank-biserial
effect size) are computed at the subject level from within-subject medians.

## Important note

This model is intentionally a sensor-space EEG model. `THAL` is a latent attention node
used for modeling and not a measured thalamic signal. Reported classifier metrics
reflect cross-validated performance on this STEW cohort; they are not evidence of
external validity on other datasets or populations.

## Figure caption block (paper-ready)

Figure A. Rest versus high-workload ASI-EEG across subjects. Paired subject-level values are connected across conditions, with rest shown in blue and high-workload in orange. The median ASI-EEG increased from 0.513 at rest to 0.531 during high workload, indicating a consistent within-subject rise in the composite EEG index.

Figure B. Within-subject distribution of the ASI-EEG change. Histogram of ΔASI-EEG = ASI-EEG(high) − ASI-EEG(rest) across 48 subjects, overlaid with the median shift and zero-reference line. The positive median change is consistent with a condition effect, supporting H1 in the current STEW dataset.
