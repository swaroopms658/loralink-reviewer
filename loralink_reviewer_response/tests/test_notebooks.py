"""Static structure checks for the sharded Colab experiment notebooks.

Notebooks are DATA -- none is executed here. We only assert their shape with
``nbformat``.
"""
import pathlib

import nbformat
import pytest

NBDIR = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
NAMES = ["00_setup_smoke", "01_stat_validation", "02_task_quality", "02b_convergence",
         "03_alt_scheduling", "04_scalability_sim", "05_network_netem"]


def _src(name):
    nb = nbformat.read(NBDIR / f"{name}.ipynb", as_version=4)
    return nb, "\n".join(c.source for c in nb.cells)


@pytest.mark.parametrize("name", NAMES)
def test_notebook_shape(name):
    nb, src = _src(name)
    assert len(nb.cells) == 5
    assert "WALL_BUDGET_MIN" in src and "budget_left()" in src
    assert "ACCOUNT_TAG" in src
    assert "SHARD" in src
    assert "files.download" in src
    assert "run_cluster(" in src
    assert "PER_RUN_ESTIMATE" in src
    assert "num_samples=100" not in src


@pytest.mark.parametrize("name", NAMES)
def test_walltime_guard_in_body(name):
    nb, _ = _src(name)
    body = nb.cells[3].source
    assert "budget_left() < PER_RUN_ESTIMATE" in body
    assert "budget exhausted" in body


@pytest.mark.parametrize("name", NAMES)
def test_failures_are_not_counted_as_successes(name):
    """The manifest is a provenance record: a crashed run must not read as done.

    NB04/NB05 previously did `DONE += 1` in their except branch, so a sweep where
    every cell failed still reported "done 12/12" and shipped a manifest saying so.
    """
    import ast

    nb, _ = _src(name)
    tree = ast.parse(nb.cells[3].source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        bumped = {
            t.target.id for t in ast.walk(node)
            if isinstance(t, ast.AugAssign) and isinstance(t.target, ast.Name)
        }
        assert "DONE" not in bumped, f"{name}: failure handler increments DONE"


@pytest.mark.parametrize("name", NAMES)
def test_manifest_reports_failures_and_files(name):
    nb, _ = _src(name)
    last = nb.cells[-1].source
    assert '"failed": FAILED' in last, name
    assert '"succeeded": DONE' in last, name
    assert "result_files" in last, name


@pytest.mark.parametrize("name", NAMES)
def test_download_cell_globs_summary(name):
    nb, _ = _src(name)
    last = nb.cells[-1].source
    assert "results_*_" in last and ".csv" in last
    assert ".summary.csv" in last


def test_02_uses_phi_and_others_125m():
    for name in ("02_task_quality", "02b_convergence"):
        assert "microsoft/phi-1_5" in _src(name)[1]
    for name in ("01_stat_validation", "03_alt_scheduling",
                 "04_scalability_sim", "05_network_netem"):
        assert "gpt-neo-125M" in _src(name)[1]


def test_02_eval_uses_limit_not_holdout():
    _, src = _src("02_task_quality")
    # Eval runs in a child process so the kernel does not hold a 1.3B model
    # while the next arm spawns three more model-loading processes.
    assert "evaluate_adapter_subprocess(" in src
    assert "limit=200" in src
    eval_line = next(ln for ln in src.splitlines()
                     if "evaluate_adapter_subprocess(" in ln)
    assert "eval_holdout" not in eval_line          # removed from evaluate_adapter
    assert "eval_holdout=200" in src                # still passed to run_cluster


def test_03_handles_partition_infeasible():
    _, src = _src("03_alt_scheduling")
    assert "except (RuntimeError, OSError)" in src
    assert "PartitionInfeasible" in src            # only a real infeasible logs the row
    assert "not infeasible" in src                 # other crashes are skipped, not recorded
    assert 'note": "infeasible"' in src or "note='infeasible'" in src
    assert "SUMMARY_COLUMNS" in src


def test_00_smoke_asserts_loss_rows():
    _, src = _src("00_setup_smoke")
    assert "SMOKE PASS" in src and "SMOKE FAIL" in src
    assert ">= 3" in src


def test_build_notebooks_idempotent():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_notebooks", NBDIR / "build_notebooks.py")
    bn = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bn)
    bn.main()
    bn.main()  # second run must overwrite cleanly
    for name in NAMES:
        p = NBDIR / f"{name}.ipynb"
        assert p.exists()
        nb = nbformat.read(p, as_version=4)
        assert nb.nbformat == 4
        assert len(nb.cells) == 5
