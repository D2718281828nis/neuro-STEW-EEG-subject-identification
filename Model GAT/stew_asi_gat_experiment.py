#!/usr/bin/env python3
"""Paired STEW rest/high ASI-EEG experiment with a latent THAL attention node.

This is the working implementation from the external analysis bundle, adapted to
run locally inside this repo. It deliberately distinguishes measured sensor-space
EEG from the latent model variable THAL.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, optimize, signal, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold


FS = 128
WINDOW_SECONDS = 4
STEP_SECONDS = 2
WINDOW_SAMPLES = FS * WINDOW_SECONDS
STEP_SAMPLES = FS * STEP_SECONDS

CHANNELS = [
    "AF3", "F7", "F3", "FC5", "T7", "P7", "O1",
    "O2", "P8", "T8", "FC6", "F4", "F8", "AF4",
]

REGIONS = {
    "PREFRONTAL": ["AF3", "AF4"],
    "FRONTAL": ["F7", "F3", "F4", "F8"],
    "FRONTOCENTRAL": ["FC5", "FC6"],
    "TEMPORAL": ["T7", "T8"],
    "PARIETAL": ["P7", "P8"],
    "OCCIPITAL": ["O1", "O2"],
}

BANDS = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}
BAND_NAMES = list(BANDS)

FRONTAL_CHANNELS = ["AF3", "F7", "F3", "FC5", "FC6", "F4", "F8", "AF4"]
POSTERIOR_CHANNELS = ["P7", "O1", "O2", "P8"]


@dataclass
class RecordingFeatures:
    subject: int
    condition: str
    relative_power: np.ndarray
    node_log_power: np.ndarray
    theta_wpli: np.ndarray
    asi_raw: np.ndarray


def _channel_indices(names: Iterable[str]) -> list[int]:
    return [CHANNELS.index(name) for name in names]


def node_mapping_frame() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    region_for_channel = {
        channel: region
        for region, members in REGIONS.items()
        for channel in members
    }
    for index, channel in enumerate(CHANNELS):
        rows.append({
            "node": f"CTX_{index}",
            "node_type": "measured_channel",
            "electrodes": channel,
            "region": region_for_channel[channel].lower(),
        })
    return pd.DataFrame(rows)


def _band_mask(frequencies: np.ndarray, low: float, high: float, *, last: bool) -> np.ndarray:
    if last:
        return (frequencies >= low) & (frequencies <= high)
    return (frequencies >= low) & (frequencies < high)


def _window_view(data: np.ndarray) -> np.ndarray:
    starts = np.arange(0, len(data) - WINDOW_SAMPLES + 1, STEP_SAMPLES)
    return np.stack([data[start:start + WINDOW_SAMPLES] for start in starts], axis=0)


def preprocess_eeg(data: np.ndarray) -> np.ndarray:
    data = signal.detrend(data.astype(np.float64), axis=0, type="linear")
    notch_b, notch_a = signal.iirnotch(50.0, 30.0, fs=FS)
    data = signal.filtfilt(notch_b, notch_a, data, axis=0)
    sos = signal.butter(4, (1.0, 45.0), btype="bandpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, data, axis=0)


def relative_band_power(windows: np.ndarray) -> np.ndarray:
    frequencies, psd = signal.welch(
        windows,
        fs=FS,
        window="hann",
        nperseg=256,
        noverlap=128,
        detrend="constant",
        scaling="density",
        axis=1,
    )
    total_mask = (frequencies >= 1.0) & (frequencies <= 45.0)
    total = integrate.trapezoid(psd[:, total_mask, :], frequencies[total_mask], axis=1)
    powers = []
    for band_index, (low, high) in enumerate(BANDS.values()):
        mask = _band_mask(frequencies, low, high, last=band_index == len(BANDS) - 1)
        powers.append(integrate.trapezoid(psd[:, mask, :], frequencies[mask], axis=1))
    absolute = np.stack(powers, axis=-1)
    return absolute / np.maximum(total[:, :, None], 1e-12)


def theta_wpli(windows: np.ndarray) -> np.ndarray:
    offsets = range(0, WINDOW_SAMPLES - FS + 1, FS // 2)
    segments = np.stack([windows[:, offset:offset + FS, :] for offset in offsets], axis=1)
    segments = segments - segments.mean(axis=2, keepdims=True)
    segments = segments * signal.windows.hann(FS, sym=False)[None, None, :, None]
    spectrum = np.fft.rfft(segments, axis=2)
    frequencies = np.fft.rfftfreq(FS, d=1.0 / FS)
    theta = (frequencies >= 4.0) & (frequencies < 8.0)

    pair_values = []
    for front in _channel_indices(FRONTAL_CHANNELS):
        for back in _channel_indices(POSTERIOR_CHANNELS):
            cross_imag = np.imag(spectrum[:, :, theta, front] * np.conj(spectrum[:, :, theta, back]))
            numerator = np.abs(cross_imag.sum(axis=1))
            denominator = np.abs(cross_imag).sum(axis=1)
            pair_values.append((numerator / np.maximum(denominator, 1e-12)).mean(axis=1))
    return np.median(np.stack(pair_values, axis=1), axis=1)


def build_node_log_power(relative_power: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(relative_power, 1e-12))


def raw_asi_components(relative_power: np.ndarray, wpli: np.ndarray) -> np.ndarray:
    theta_i, alpha_i, beta_i = (BAND_NAMES.index(name) for name in ("theta", "alpha", "beta"))
    frontal = relative_power[:, _channel_indices(FRONTAL_CHANNELS), :]
    posterior = relative_power[:, _channel_indices(POSTERIOR_CHANNELS), :]

    engagement = np.log(
        np.maximum(frontal[:, :, beta_i].mean(axis=1), 1e-12)
        / np.maximum(
            frontal[:, :, theta_i].mean(axis=1) + frontal[:, :, alpha_i].mean(axis=1),
            1e-12,
        )
    )
    alpha_gating = -np.log(np.maximum(posterior[:, :, alpha_i].mean(axis=1), 1e-12))

    channel_engagement = relative_power[:, :, beta_i] / np.maximum(
        relative_power[:, :, theta_i] + relative_power[:, :, alpha_i], 1e-12
    )
    distribution = channel_engagement / np.maximum(channel_engagement.sum(axis=1, keepdims=True), 1e-12)
    entropy = -(distribution * np.log(np.maximum(distribution, 1e-12))).sum(axis=1)
    selectivity = 1.0 - entropy / math.log(len(CHANNELS))
    return np.column_stack([engagement, alpha_gating, wpli, selectivity])


def extract_recording(path: Path) -> RecordingFeatures:
    match = re.fullmatch(r"sub(\d{2})_(lo|hi)\.txt", path.name)
    if not match:
        raise ValueError(f"Unexpected STEW filename: {path.name}")
    subject = int(match.group(1))
    condition = "rest" if match.group(2) == "lo" else "high"
    data = np.loadtxt(path)
    if data.shape != (19200, len(CHANNELS)):
        raise ValueError(f"{path}: expected (19200, 14), got {data.shape}")
    windows = _window_view(preprocess_eeg(data))
    relative_power = relative_band_power(windows)
    wpli = theta_wpli(windows)
    return RecordingFeatures(
        subject=subject,
        condition=condition,
        relative_power=relative_power,
        node_log_power=build_node_log_power(relative_power),
        theta_wpli=wpli,
        asi_raw=raw_asi_components(relative_power, wpli),
    )


def _robust_location_scale(values: np.ndarray, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(values, axis=axis)
    mad = np.median(np.abs(values - np.expand_dims(center, axis=axis)), axis=axis)
    scale = 1.4826 * mad
    standard = np.std(values, axis=axis, ddof=1)
    scale = np.where(scale > 1e-6, scale, np.maximum(standard, 1e-6))
    return center, scale


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def baseline_normalize(records: list[RecordingFeatures]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    by_key = {(record.subject, record.condition): record for record in records}
    metadata: list[dict[str, object]] = []
    node_features: list[np.ndarray] = []
    asi_components: list[np.ndarray] = []
    asi_values: list[np.ndarray] = []

    for subject in sorted({record.subject for record in records}):
        rest = by_key[(subject, "rest")]
        component_center, component_scale = _robust_location_scale(rest.asi_raw)
        node_center, node_scale = _robust_location_scale(rest.node_log_power)

        for condition in ("rest", "high"):
            record = by_key[(subject, condition)]
            component_z = (record.asi_raw - component_center) / component_scale
            component_scores = _sigmoid(np.clip(component_z, -6.0, 6.0))
            asi = component_scores.mean(axis=1)

            node_z = (record.node_log_power - node_center) / node_scale
            node_z[:, :, BAND_NAMES.index("alpha")] *= -1.0
            node_z = np.clip(node_z, -6.0, 6.0)

            for window_index in range(len(asi)):
                metadata.append({
                    "subject": subject,
                    "condition": condition,
                    "window": window_index,
                    "start_seconds": window_index * STEP_SECONDS,
                })
            node_features.append(node_z)
            asi_components.append(component_scores)
            asi_values.append(asi)

    return (
        pd.DataFrame(metadata),
        np.concatenate(node_features),
        np.concatenate(asi_components),
        np.concatenate(asi_values),
    )


def _attention_forward(parameters: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_features = x.shape[-1]
    query = parameters[:n_features]
    classifier = parameters[n_features:2 * n_features]
    bias = parameters[-1]
    scores = np.einsum("nif,f->ni", x, query)
    scores = scores - scores.max(axis=1, keepdims=True)
    attention = np.exp(scores)
    attention /= attention.sum(axis=1, keepdims=True)
    pooled = np.einsum("ni,nif->nf", attention, x)
    probability = _sigmoid(pooled @ classifier + bias)
    return probability, attention, pooled


def _attention_loss_gradient(
    parameters: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
) -> tuple[float, np.ndarray]:
    probability, attention, pooled = _attention_forward(parameters, x)
    n_features = x.shape[-1]
    query = parameters[:n_features]
    classifier = parameters[n_features:2 * n_features]
    eps = 1e-9
    loss = -np.mean(y * np.log(probability + eps) + (1.0 - y) * np.log(1.0 - probability + eps))
    loss += 0.5 * l2 * (query @ query + classifier @ classifier)

    d_logit = (probability - y) / len(y)
    grad_classifier = pooled.T @ d_logit + l2 * classifier
    grad_bias = d_logit.sum()
    d_pooled = d_logit[:, None] * classifier[None, :]
    d_scores = attention * np.einsum("nf,nif->ni", d_pooled, x - pooled[:, None, :])
    grad_query = np.einsum("ni,nif->f", d_scores, x) + l2 * query
    gradient = np.concatenate([grad_query, grad_classifier, [grad_bias]])
    return float(loss), gradient


def fit_thal_attention(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_features = x.shape[-1]
    mean_features = x.mean(axis=1)
    logistic = LogisticRegression(C=10.0, max_iter=1000, random_state=seed)
    logistic.fit(mean_features, y)
    base_classifier = logistic.coef_.ravel()
    base_bias = float(logistic.intercept_[0])

    best = None
    for restart in range(3):
        initial = np.concatenate([
            rng.normal(0.0, 0.05 if restart else 0.0, size=n_features),
            base_classifier,
            [base_bias],
        ])
        result = optimize.minimize(
            _attention_loss_gradient,
            initial,
            args=(x, y, 1e-2),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 300, "ftol": 1e-10},
        )
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    return best.x


def cross_validated_thal(
    metadata: pd.DataFrame,
    node_features: np.ndarray,
    seed: int,
    splits: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    keep = metadata["window"].to_numpy() % 2 == 0
    model_meta = metadata.loc[keep].reset_index(drop=True)
    x = node_features[keep]
    y = (model_meta["condition"].to_numpy() == "high").astype(float)
    groups = model_meta["subject"].to_numpy()

    probabilities = np.zeros(len(x))
    attentions = np.zeros((len(x), len(CHANNELS)))
    fold_ids = np.zeros(len(x), dtype=int)
    splitter = GroupKFold(n_splits=splits)
    for fold, (train_index, test_index) in enumerate(splitter.split(x, y, groups), start=1):
        parameters = fit_thal_attention(x[train_index], y[train_index], seed + fold)
        probability, attention, _ = _attention_forward(parameters, x[test_index])
        probabilities[test_index] = probability
        attentions[test_index] = attention
        fold_ids[test_index] = fold

    output = model_meta.copy()
    output["fold"] = fold_ids
    output["probability_high"] = probabilities
    for channel_index, channel in enumerate(CHANNELS):
        output[f"attention_{channel.lower()}"] = attentions[:, channel_index]
    entropy = -(attentions * np.log(np.maximum(attentions, 1e-12))).sum(axis=1)
    output["relay_effective_span"] = np.exp(entropy)
    output["relay_penalty"] = (output["relay_effective_span"] - 1.0) / (len(CHANNELS) - 1.0)

    recording_probability = output.groupby(["subject", "condition"])["probability_high"].median()
    recording_y = np.array([condition == "high" for _, condition in recording_probability.index], dtype=int)
    metrics = {
        "recording_accuracy": float(accuracy_score(recording_y, recording_probability.to_numpy() >= 0.5)),
        "recording_auc": float(roc_auc_score(recording_y, recording_probability.to_numpy())),
        "window_accuracy": float(accuracy_score(y, probabilities >= 0.5)),
        "window_auc": float(roc_auc_score(y, probabilities)),
    }
    return output, metrics


def _paired_rank_biserial(delta: np.ndarray) -> float:
    nonzero = delta[delta != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def paired_summary(low: np.ndarray, high: np.ndarray, rng: np.random.Generator, bootstrap: int) -> dict[str, float]:
    delta = high - low
    boot = np.empty(bootstrap)
    for index in range(bootstrap):
        sample = rng.integers(0, len(delta), size=len(delta))
        boot[index] = np.median(delta[sample])
    two_sided = stats.wilcoxon(high, low, alternative="two-sided", zero_method="wilcox")
    greater = stats.wilcoxon(high, low, alternative="greater", zero_method="wilcox")
    return {
        "rest_median": float(np.median(low)),
        "high_median": float(np.median(high)),
        "median_delta": float(np.median(delta)),
        "median_delta_ci95_low": float(np.quantile(boot, 0.025)),
        "median_delta_ci95_high": float(np.quantile(boot, 0.975)),
        "mean_delta": float(np.mean(delta)),
        "cohen_dz": float(np.mean(delta) / np.std(delta, ddof=1)),
        "rank_biserial": _paired_rank_biserial(delta),
        "wilcoxon_two_sided_p": float(two_sided.pvalue),
        "wilcoxon_greater_p": float(greater.pvalue),
        "n": int(len(delta)),
    }


def spearman_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    bootstrap: int,
) -> dict[str, float]:
    result = stats.spearmanr(x, y)
    boot = []
    for _ in range(bootstrap):
        sample = rng.integers(0, len(x), size=len(x))
        value = stats.spearmanr(x[sample], y[sample]).statistic
        if np.isfinite(value):
            boot.append(value)
    return {
        "spearman_rho": float(result.statistic),
        "spearman_p": float(result.pvalue),
        "spearman_ci95_low": float(np.quantile(boot, 0.025)),
        "spearman_ci95_high": float(np.quantile(boot, 0.975)),
        "n": int(len(x)),
    }


def save_h1_proof_plots(subject_condition: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    wide = subject_condition.pivot(index="subject", columns="condition")
    rest = wide[("asi_eeg", "rest")].to_numpy()
    high = wide[("asi_eeg", "high")].to_numpy()
    delta = high - rest

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx in range(len(rest)):
        ax.plot([0, 1], [rest[idx], high[idx]], color="0.7", linewidth=1, alpha=0.5)
    rest_handle = ax.scatter([0] * len(rest), rest, color="tab:blue", s=18, alpha=0.8, label="rest")
    high_handle = ax.scatter([1] * len(high), high, color="tab:orange", s=18, alpha=0.8, label="high")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["rest", "high"])
    ax.set_ylabel("ASI-EEG")
    ax.set_title("H1: paired rest vs high ASI-EEG across subjects")
    ax.legend(handles=[rest_handle, high_handle], loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "h1_paired_asi_eeg.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    counts, bins, _ = ax.hist(delta, bins=12, color="tab:green", edgecolor="black", alpha=0.8, label="subject ΔASI")
    ax.axvline(np.median(delta), color="black", linestyle="--", linewidth=2, label=f"median Δ = {np.median(delta):.3f}")
    ax.axvline(0.0, color="red", linestyle="-", linewidth=1.5, label="zero")
    ax.set_xlabel("Δ ASI-EEG (high - rest)")
    ax.set_ylabel("count")
    ax.set_title("H1: distribution of within-subject ASI change")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "h1_delta_distribution.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("Model/results"))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()

    files = sorted(args.dataset.glob("sub??_*.txt"))
    if len(files) != 96:
        raise ValueError(f"Expected 96 STEW EEG files, found {len(files)} in {args.dataset}")
    records = [extract_recording(path) for path in files]
    metadata, node_features, component_scores, asi = baseline_normalize(records)

    component_names = ["engagement", "alpha_gating", "phase_organization", "spatial_selectivity"]
    windows = metadata.copy()
    for index, name in enumerate(component_names):
        windows[name] = component_scores[:, index]
    windows["asi_eeg"] = asi

    thal_windows, gat_metrics = cross_validated_thal(metadata, node_features, args.seed, args.splits)
    thal_windows = thal_windows.merge(
        windows[["subject", "condition", "window", "asi_eeg"]],
        on=["subject", "condition", "window"],
        how="left",
        validate="one_to_one",
    )
    thal_windows["asi_projected"] = np.clip(
        thal_windows["asi_eeg"] - 0.15 * thal_windows["relay_penalty"], 0.0, 1.0
    )

    subject_condition = windows.groupby(["subject", "condition"], as_index=False)[
        component_names + ["asi_eeg"]
    ].median()
    thal_subject_condition = thal_windows.groupby(["subject", "condition"], as_index=False).agg(
        asi_projected=("asi_projected", "median"),
        relay_effective_span=("relay_effective_span", "median"),
        probability_high=("probability_high", "median"),
    )
    subject_condition = subject_condition.merge(
        thal_subject_condition,
        on=["subject", "condition"],
        validate="one_to_one",
    )

    wide = subject_condition.pivot(index="subject", columns="condition")
    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {
        "design": {
            "subjects": 48,
            "recordings": 96,
            "sampling_hz": FS,
            "window_seconds": WINDOW_SECONDS,
            "step_seconds": STEP_SECONDS,
            "ctx_nodes": 14,
            "measured_channel_nodes": 14,
            "regional_composite_nodes": 0,
            "thal_neighbors": 14,
            "thal_status": "latent GAT readout; not measured thalamic EEG",
            "baseline": "within-subject rest median and MAD",
        },
        "gat_condition_classifier": gat_metrics,
        "paired": {},
    }
    for name in component_names + ["asi_eeg", "asi_projected", "relay_effective_span"]:
        summary["paired"][name] = paired_summary(
            wide[(name, "rest")].to_numpy(),
            wide[(name, "high")].to_numpy(),
            rng,
            args.bootstrap,
        )

    save_h1_proof_plots(subject_condition, args.output)

    attention_columns = [column for column in thal_windows if column.startswith("attention_")]
    attention_summary = thal_windows.groupby("condition", as_index=False)[attention_columns].mean()
    regional_attention = thal_windows[["condition"]].copy()
    for region, members in REGIONS.items():
        regional_attention[f"attention_{region.lower()}"] = thal_windows[
            [f"attention_{channel.lower()}" for channel in members]
        ].sum(axis=1)
    regional_attention_summary = regional_attention.groupby("condition", as_index=False).mean()

    args.output.mkdir(parents=True, exist_ok=True)
    node_mapping_frame().to_csv(args.output / "ctx_node_mapping.csv", index=False)
    windows.to_csv(args.output / "window_asi.csv", index=False)
    thal_windows.to_csv(args.output / "thal_cross_validated_windows.csv", index=False)
    subject_condition.to_csv(args.output / "subject_condition_metrics.csv", index=False)
    attention_summary.to_csv(args.output / "thal_attention_by_condition.csv", index=False)
    regional_attention_summary.to_csv(args.output / "thal_attention_by_region.csv", index=False)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
