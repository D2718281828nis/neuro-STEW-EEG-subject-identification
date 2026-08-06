from __future__ import annotations

import pytest
from _helpers import load_module


@pytest.fixture(scope="session")
def eeg_config_module():
    return load_module("eeg_config", "eeg_config.py")


@pytest.fixture(scope="session")
def subject_ident_module():
    return load_module("subject_ident_pipeline", "Model Subject Ident/stewSubjectsIdentification.py")


@pytest.fixture(scope="session")
def kuramoto_module():
    return load_module("kuramoto_gnn_pipeline", "Model KuramotoGNN/kuramoto_gnn_stew.py")


@pytest.fixture(scope="session")
def gat_module():
    return load_module("gat_pipeline", "Model GAT/stew_asi_gat_experiment.py")
