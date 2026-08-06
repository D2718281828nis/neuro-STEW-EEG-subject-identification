# Model Subject Ident

This folder contains a classical EEG-based subject identification pipeline built on the STEW dataset.

## What this model does

The pipeline performs:

- bandpass filtering and despiking of 14-channel EEG signals
- sliding-window segmentation of each subject recording (window/step duration shared
  with the other two pipelines via `eeg_config.py`)
- extraction of statistical, spectral, entropy, and fractal features from each channel
  (via [`antropy`](https://github.com/raphaelvallat/antropy) for entropy/fractal terms)
- feature aggregation into a single feature table with window metadata (subject,
  condition, recording, window index, start time)
- training a small Keras artificial neural network to classify window features by subject
- visualization of classification performance with training curves and confusion matrices

## Key features extracted

- PSD statistics: mean and standard deviation of the Welch power spectral density
- amplitude statistics: mean, variance, range, skewness, kurtosis
- entropy metrics: permutation entropy, spectral entropy, SVD entropy, approximate entropy, sample entropy
- fractal dimension metrics: Petrosian, Katz, Higuchi
- detrended fluctuation analysis

## Usage

From the repository root, with `dataset/` populated:

```bash
python "Model Subject Ident/stewSubjectsIdentification.py" --extract-features
```

Subsequent runs can reuse the extracted feature table instead of recomputing it:

```bash
python "Model Subject Ident/stewSubjectsIdentification.py"
```

Useful CLI options (see `--help` for the full list): `--dataset-dir`, `--feature-csv`,
`--output-dir`, `--seed`, `--epochs`, `--batch-size`, `--window-size`, `--step-size`,
`--split-count`.

## Evaluation methodology

Subject identity is the classification target, so leave-subject-out validation does not
apply here. Instead, each subject's windows are sorted in time and split into
`--split-count` (default 3) contiguous, non-overlapping temporal blocks; the first block
of every subject goes to training, the second to validation, and the remainder to test.
This keeps overlapping/adjacent windows from crossing a partition boundary while still
letting every subject appear in every partition. Feature scaling (`StandardScaler`) is
fit on the training partition only. All reported metrics are **window-level**, not
recording- or subject-level.

## Outputs

Written under `--output-dir` (default `Model Subject Ident/out/`):

- `metrics.json` — validation/test accuracy, macro precision/recall/F1, confusion matrices
- `summary.json` — effective CLI configuration, dataset/subject/window counts, dependency versions, and evaluation metrics
- `training_validation_acc.png` / `training_validation_loss.png` — ANN training curves
- `confusion_matrix.png` — test-set confusion matrix

The extracted feature table is written to `--feature-csv` (default `Model Subject Ident/extractedFeatures.csv`).

## Notes and limitations

This pipeline is an exploratory baseline for subject identification using classical EEG
features. Because partitions are temporal blocks within-subject rather than held-out
recordings or subjects, reported accuracy reflects how well the model recognizes a
subject's later windows after seeing earlier windows from the same recording session —
not generalization to a new recording session or a new day.
