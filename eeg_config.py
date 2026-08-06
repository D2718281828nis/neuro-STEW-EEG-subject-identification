from __future__ import annotations

import random
from typing import Final

import numpy as np
import torch

FS: Final[int] = 128
WINDOW_SECONDS: Final[float] = 4.0
STEP_SECONDS: Final[float] = 2.0
WINDOW_SAMPLES: Final[int] = int(FS * WINDOW_SECONDS)
STEP_SAMPLES: Final[int] = int(FS * STEP_SECONDS)

CHANNELS: Final[tuple[str, ...]] = (
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
)

REGIONS: Final[dict[str, list[str]]] = {
    "PREFRONTAL": ["AF3", "AF4"],
    "FRONTAL": ["F7", "F3", "F4", "F8"],
    "FRONTOCENTRAL": ["FC5", "FC6"],
    "TEMPORAL": ["T7", "T8"],
    "PARIETAL": ["P7", "P8"],
    "OCCIPITAL": ["O1", "O2"],
}

BANDS: Final[dict[str, tuple[float, float]]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

BAND_NAMES: Final[tuple[str, ...]] = tuple(BANDS.keys())
FRONTAL_CHANNELS: Final[tuple[str, ...]] = ("AF3", "F7", "F3", "FC5", "FC6", "F4", "F8", "AF4")
POSTERIOR_CHANNELS: Final[tuple[str, ...]] = ("P7", "O1", "O2", "P8")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
