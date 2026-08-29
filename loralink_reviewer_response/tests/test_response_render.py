"""Task 13 - render_response substitution + shipped-doc invariants."""
import json
import re

from loralink_reviewer_response.aggregate import render_response


def test_placeholders_filled(tmp_path):
    sj = tmp_path / "s.json"
    sj.write_text(json.dumps({
        "quality": {"e2e": {"delta_ppl_on_minus_off": 0.4}},
        "a": {"b": {"c": 1.25}},
    }))
    tpl = tmp_path / "tpl.md"
    tpl.write_text("ppl {{quality.e2e.delta_ppl_on_minus_off}} / {{a.b.c}}")
    out = tmp_path / "out.md"
    render_response(str(sj), str(tpl), str(out))
    txt = out.read_text()
    assert txt == "ppl 0.4 / 1.25"
    assert "{{" not in txt


def test_readme_and_template_exist(pkg_dir):
    assert (pkg_dir / "README.md").stat().st_size > 500
    t = (pkg_dir / "RESPONSE_ABHAY_NIKHIL.md").read_text(encoding="utf-8")
    assert t.count("{{") >= 6
    low = t.lower()
    assert "[published" in low
    assert "loopback" in low
    assert "[ours]" in t


def test_response_covers_8_concerns(pkg_dir):
    t = (pkg_dir / "RESPONSE_ABHAY_NIKHIL.md").read_text(encoding="utf-8")
    headings = sorted(re.findall(r"(?m)^##\s+Concern\s+([1-8])\b", t))
    assert headings == [str(i) for i in range(1, 9)]
