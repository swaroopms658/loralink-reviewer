import subprocess, sys

SRC_FILES = ["main.py", "device_manager.py", "compression_engine.py",
             "benchmarking.py", "data_loader.py"]


def test_checksums_verify_clean(pkg_dir):
    # --verify only: the committed SHA256SUMS must already match the tree
    # (notebook cell 1 runs the same bare --verify -- no --update to mask a drift).
    r = subprocess.run([sys.executable, str(pkg_dir / "patch" / "checksums.py"), "--verify"])
    assert r.returncode == 0


def test_patch_files_exist_and_nonempty(pkg_dir):
    for f in SRC_FILES:
        p = pkg_dir / "patch" / f"{f}.patch"
        assert p.exists() and p.stat().st_size > 0


def test_patches_mention_only_allowed_changes(pkg_dir):
    banned = ["optimizer", "adamw", "lr=", "learning_rate", "loss =",
              "backward()", "sparsify_by_magnitude(", "quantize_to_int8("]
    for f in ["compression_engine.py", "device_manager.py"]:
        txt = (pkg_dir / "patch" / f"{f}.patch").read_text(encoding="utf-8")
        added = "\n".join(
            l for l in txt.splitlines()
            if l.startswith("+") and not l.startswith("+++")
        ).lower()
        for b in banned:
            assert b not in added, (f, b)
