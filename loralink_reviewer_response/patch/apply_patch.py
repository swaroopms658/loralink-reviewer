"""Apply the reviewer-response source patches to a fresh repo copy.

In the Colab flow the repo IS already the patched checkout (we ship the
patched sources in this package / branch). So apply_patch degrades to an
integrity check: it runs checksums.py --verify and exits its code.
"""
import argparse, subprocess, sys, pathlib
PATCH_DIR = pathlib.Path(__file__).resolve().parent
SRC_ROOT = PATCH_DIR.parents[1]
FILES = ["main.py", "device_manager.py", "compression_engine.py",
         "benchmarking.py", "data_loader.py"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    dst = pathlib.Path(a.repo)
    r = subprocess.run([sys.executable, str(PATCH_DIR / "checksums.py"), "--verify"])
    sys.exit(r.returncode)

if __name__ == "__main__":
    main()
