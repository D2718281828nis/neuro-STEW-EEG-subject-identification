# STEW ASI + latent THAL attention model

This folder contains a verified experiment that runs on the local STEW dataset in this repository.

## What it does

- reads the 14-channel STEW EEG recordings under `dataset/`
- extracts per-window band-power and phase features
- computes a subject-level rest-normalized ASI-EEG index
- fits a latent THAL graph-attention readout over the 14 measured CTX nodes
- evaluates rest vs high workload with paired statistics and cross-validated classification

## Run

```bash
cd /Users/denmacair/Documents/code_base/neuro-STEW-EEG-subject-identification
python Model/stew_asi_gat_experiment.py --dataset dataset --output Model/results --bootstrap 200
```

The run produces files such as:

- `Model/results/summary.json`
- `Model/results/ctx_node_mapping.csv`
- `Model/results/subject_condition_metrics.csv`

## Important note

This model is intentionally a sensor-space EEG model. `THAL` is a latent attention node used for modeling and not a measured thalamic signal.

## Figure caption block (paper-ready)

Figure A. Rest versus high-workload ASI-EEG across subjects. Paired subject-level values are connected across conditions, with rest shown in blue and high-workload in orange. The median ASI-EEG increased from 0.513 at rest to 0.531 during high workload, indicating a consistent within-subject rise in the composite EEG index.

Figure B. Within-subject distribution of the ASI-EEG change. Histogram of ΔASI-EEG = ASI-EEG(high) − ASI-EEG(rest) across 48 subjects, overlaid with the median shift and zero-reference line. The positive median change is consistent with a condition effect, supporting H1 in the current STEW dataset.
