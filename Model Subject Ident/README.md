# Model Subject Ident

This folder contains a classical EEG-based subject identification pipeline built on the STEW dataset.

## What this model does

The pipeline performs:

- bandpass filtering and artifact-aware preprocessing of 14-channel EEG signals
- sliding-window segmentation of each subject recording
- extraction of statistical, spectral, entropy, and fractal features from each channel
- feature aggregation into a single feature table
- model training and evaluation using traditional machine learning techniques
- visualization of classification performance with training curves and confusion matrices

## Key features extracted

- PSD statistics: mean and standard deviation of the Welch power spectral density
- amplitude statistics: mean, variance, range, skewness, kurtosis
- entropy metrics: permutation entropy, spectral entropy, SVD entropy, approximate entropy, sample entropy
- fractal dimension metrics: Petrosian, Katz, Higuchi
- detrended fluctuation analysis

## Usage

From the repository root:

```bash
python "Model Subject Ident/stewSubjectsIdentification.py"
```

## Outputs

Typical outputs include:

- `extractedFeatures.csv` — extracted feature dataset
- `training_validation_acc.png` — model accuracy curves
- `training_validation_loss.png` — model loss curves
- `confusion_matrix.png` — classification confusion matrix

## Notes

This pipeline is designed as an exploratory baseline for subject identification using classical EEG features. It can be extended with additional feature selection, cross-validation, and neural network classifier workflows.
