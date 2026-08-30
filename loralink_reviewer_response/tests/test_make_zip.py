"""Task 14 - MAKE_ZIP.py builds a clean deliverable zip."""
import zipfile

from loralink_reviewer_response.MAKE_ZIP import build_zip


def test_make_zip_builds(tmp_path):
    out = build_zip(str(tmp_path / "deliverable.zip"))
    assert out.exists() and out.stat().st_size > 0

    names = zipfile.ZipFile(out).namelist()

    def present(suffix):
        return any(n == suffix or n.endswith("/" + suffix) for n in names)

    for want in ("RESPONSE_ABHAY_NIKHIL.md", "README.md", "HOW_TO_RUN.txt",
                 "VERIFICATION.md", "aggregate.py",
                 "notebooks/99_aggregate_report.ipynb", "patch/SHA256SUMS"):
        assert present(want), want

    for n in names:
        assert "__pycache__" not in n, n
        assert not n.endswith(".pyc"), n
        assert not n.endswith(".zip"), n
        assert not (n.startswith("loralink_reviewer_response/results/")
                    and n.endswith(".csv")), n
        assert not (n.startswith("loralink_reviewer_response/figures/")
                    and n.endswith(".png")), n

    # empty output dirs are kept via a .gitkeep
    assert "loralink_reviewer_response/results/.gitkeep" in names
    assert "loralink_reviewer_response/figures/.gitkeep" in names


def test_how_to_run_exists(pkg_dir):
    txt = (pkg_dir / "HOW_TO_RUN.txt").read_text(encoding="utf-8")
    assert "GH_PAT" in txt
    assert "99_aggregate_report" in txt
