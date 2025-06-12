from __future__ import annotations

from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def root_dir() -> Path:
    return Path(__file__).parents[2]


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device(0) if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture(scope="module")
def output_dir() -> Path:
    return Path('src/test/smoke/smoke_test')
