import csv
import re


def test_no_unfilled_placeholders(pkg_dir):
    txt = (pkg_dir / "baselines" / "published_baselines.csv").read_text(encoding="utf-8")
    assert "<FILL>" not in txt and "TODO" not in txt


def test_every_row_has_a_source_block(pkg_dir):
    csv_path = pkg_dir / "baselines" / "published_baselines.csv"
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    src = (pkg_dir / "baselines" / "SOURCES.md").read_text(encoding="utf-8")
    assert len(rows) >= 6
    for r in rows:
        assert r["comparable"] in {"direct", "trend", "context"}
        assert f'[{r["source_ref"]}]' in src or f'## {r["source_ref"]}' in src
        assert float(re.sub(r"[^0-9.\-]", "", r["value"]))  # value is numeric-ish


def test_sources_have_urls_and_quotes(pkg_dir):
    src = (pkg_dir / "baselines" / "SOURCES.md").read_text(encoding="utf-8")
    assert src.count("arxiv.org") + src.count("doi.org") >= 5
    assert src.count(">") >= 6  # at least one verbatim quote per source
