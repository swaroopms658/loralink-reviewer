"""Build the deliverable ``loralink_reviewer_response.zip``.

    python loralink_reviewer_response/MAKE_ZIP.py [OUT_PATH]

Zips the ``loralink_reviewer_response/`` package (and nothing else - the
notebooks ``git clone`` the full patched repo separately, see HOW_TO_RUN.txt).
Excludes generated / heavy dirs; keeps ``results/`` and ``figures/`` as empty
dirs via a synthesized ``.gitkeep``.
"""
from __future__ import annotations

import pathlib
import sys
import zipfile

PKG = pathlib.Path(__file__).resolve().parent
REPO_ROOT = PKG.parent
PKG_NAME = PKG.name

_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", "adapters", ".ipynb_checkpoints"}
_EXCLUDE_SUFFIX = {".pyc", ".pyo", ".zip"}
# dirs whose *contents* are dropped but the dir itself is kept (via .gitkeep)
_EMPTY_DIRS = {"results", "figures"}
_HOW_TO_RUN = "HOW_TO_RUN.txt"


def _included_files():
    """Yield (abs_path, arcname) for every file that belongs in the zip."""
    for path in sorted(PKG.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(PKG)
        parts = set(rel.parts)
        if parts & _EXCLUDE_DIRS:
            continue
        if path.suffix in _EXCLUDE_SUFFIX:
            continue
        if rel.parts[0] in _EMPTY_DIRS and rel.name != ".gitkeep":
            continue
        yield path, f"{PKG_NAME}/{rel.as_posix()}"


def build_zip(out_path=None) -> pathlib.Path:
    out = pathlib.Path(out_path) if out_path else REPO_ROOT / "loralink_reviewer_response.zip"
    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    how_to_run = (PKG / _HOW_TO_RUN).read_text(encoding="utf-8")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for abs_path, arcname in _included_files():
            zf.write(abs_path, arcname)
        # keep results/ and figures/ present but empty
        for d in sorted(_EMPTY_DIRS):
            if not (PKG / d / ".gitkeep").exists():
                zf.writestr(f"{PKG_NAME}/{d}/.gitkeep", "")
        # config anchor: pytest.ini lives at the repo root (not in this package),
        # so ship a copy inside the package for standalone `pytest` runs.
        if not (PKG / "pytest.ini").exists():
            zf.writestr(f"{PKG_NAME}/pytest.ini",
                        (REPO_ROOT / "pytest.ini").read_text(encoding="utf-8"))
        # a copy at the zip root so it is the first thing the user sees
        zf.writestr(_HOW_TO_RUN, how_to_run)

    names = zipfile.ZipFile(out).namelist()
    size = out.stat().st_size
    print(f"wrote {out}")
    print(f"  size:  {size:,} bytes ({size / 1024:.1f} KiB)")
    print(f"  files: {len(names)}")
    return out


def main():
    build_zip(sys.argv[1] if len(sys.argv) > 1 else None)


if __name__ == "__main__":
    main()
