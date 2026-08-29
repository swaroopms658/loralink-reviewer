# patch/ — reviewer-response source changes

These `.patch` files are the additive changes made to the five LoraLink source
files (`main.py`, `device_manager.py`, `compression_engine.py`,
`benchmarking.py`, `data_loader.py`) for the Colab reviewer-response runs. The
patched tree ships directly — the Colab notebooks clone this branch with the
changes already merged in, so nothing needs to be applied. Each
`<file>.patch` is `git diff 55e1714 -- <file>` and exists only so a reviewer
can see exactly what changed versus the paper's original code (commit
`55e1714`). `apply_patch.py --repo <dir>` therefore just runs
`checksums.py --verify`, which compares the live sources against the recorded
`SHA256SUMS` and exits non-zero on any mismatch. Regenerate the baseline with
`python checksums.py --update`.
