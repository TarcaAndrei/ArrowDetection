import shutil
import time
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def config_path() -> str:
    return 'src/configs/test.py'

@pytest.fixture(scope='module', autouse=True)
def clean(output_dir: Path):
    # setup
    if output_dir.exists():
        shutil.rmtree(output_dir)
    yield
    # teardown
    if output_dir.exists():
        # tensorboard saves event files asynchronously so a sleep is required to fully clean the output dir
        time.sleep(0.5)
        shutil.rmtree(output_dir)
