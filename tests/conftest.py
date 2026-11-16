import pytest
from src.data.synthetic import generate
from src.data.validation import read_raw


@pytest.fixture
def frames(tmp_path):
    generate(tmp_path, n=300)
    return read_raw(tmp_path)
