from __future__ import annotations

import numpy as np


def test_window_and_step_samples_derived_from_seconds(eeg_config_module):
    cfg = eeg_config_module
    assert cfg.WINDOW_SAMPLES == int(cfg.FS * cfg.WINDOW_SECONDS)
    assert cfg.STEP_SAMPLES == int(cfg.FS * cfg.STEP_SECONDS)


def test_channels_and_bands_are_consistent(eeg_config_module):
    cfg = eeg_config_module
    assert len(cfg.CHANNELS) == 14
    assert set(cfg.BAND_NAMES) == set(cfg.BANDS)
    for low, high in cfg.BANDS.values():
        assert low < high


def test_set_seed_is_reproducible(eeg_config_module):
    cfg = eeg_config_module
    cfg.set_seed(42)
    first = np.random.rand(5)
    cfg.set_seed(42)
    second = np.random.rand(5)
    assert np.array_equal(first, second)
