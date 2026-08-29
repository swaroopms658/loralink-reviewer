import pathlib
import pytest

@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    # tests/ -> loralink_reviewer_response/ -> repo root
    return pathlib.Path(__file__).resolve().parents[2]

@pytest.fixture(scope="session")
def pkg_dir(repo_root) -> pathlib.Path:
    return repo_root / "loralink_reviewer_response"
