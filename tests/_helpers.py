"""Utilities for importing pipeline scripts that live in space-containing directories."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str) -> ModuleType:
    """Load a script by file path and register it in sys.modules under `name`.

    Needed because directories such as "Model Subject Ident" contain spaces and
    are not Python packages, so a normal `import` statement cannot reach them.
    """
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_stew_recording(
    path: Path, n_samples: int, n_channels: int = 14, *, seed: int = 0, scale: float = 50.0, offset: float = 4000.0
) -> None:
    """Write a synthetic STEW-format whitespace-delimited recording file."""
    import numpy as np

    rng = np.random.default_rng(seed)
    data = rng.normal(loc=offset, scale=scale, size=(n_samples, n_channels))
    np.savetxt(path, data)
