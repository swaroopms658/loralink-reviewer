import pathlib
import pytest

@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    # tests/ -> loralink_reviewer_response/ -> repo root
    return pathlib.Path(__file__).resolve().parents[2]

@pytest.fixture(scope="session")
def pkg_dir(repo_root) -> pathlib.Path:
    return repo_root / "loralink_reviewer_response"


@pytest.fixture(scope="session")
def tokenizer():
    """A small real tokenizer; loader tests only need it to satisfy type asserts."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("gpt2")
