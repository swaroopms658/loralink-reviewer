"""Network shaping must degrade to the in-process shim, not to nothing.

Colab runs under gVisor: the `tc` binary exists and the session is root, so
`_tc_available()` said yes, but there is no `sch_netem` module and every
`tc qdisc add` exits 2. All twelve cells of NB05 failed identically and the
sweep produced no data at all.

Availability now means "the command actually worked", not "the binary is
installed".
"""
import subprocess

import pytest

from loralink_reviewer_response import cluster_launch


def _run_with_stubbed_children(monkeypatch, tmp_path, netem):
    """Drive run_cluster far enough to fix the netem mode, then bail out."""
    seen = {}

    def _stop(*a, **kw):
        seen["env"] = kw.get("env", {})
        raise KeyboardInterrupt

    # Stub the teardown too: patching subprocess.Popen would otherwise break
    # _clear_netem's internal subprocess.run, aborting the run before the mode
    # is recorded. Nothing to tear down here anyway -- tc never applied.
    monkeypatch.setattr(cluster_launch, "_clear_netem", lambda: None)
    monkeypatch.setattr(cluster_launch.subprocess, "Popen", _stop)
    csv = tmp_path / "r.csv"
    with pytest.raises(KeyboardInterrupt):
        cluster_launch.run_cluster(1, "wikitext", 0, netem=netem,
                                   results_csv=str(csv))
    mode = csv.with_name(csv.name + ".netem").read_text()
    return mode, seen.get("env", {})


def test_failing_tc_falls_back_to_the_shim(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster_launch, "_tc_available", lambda: True)

    def _boom(parts):
        raise subprocess.CalledProcessError(2, parts)

    monkeypatch.setattr(cluster_launch, "_apply_netem", _boom)

    mode, env = _run_with_stubbed_children(
        monkeypatch, tmp_path, {"delay_ms": 25, "loss_pct": 1})

    assert mode == "in-process-shim", mode
    assert env["LORALINK_NET_SHIM"] == "25,1"


def test_absent_tc_still_uses_the_shim(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster_launch, "_tc_available", lambda: False)
    mode, env = _run_with_stubbed_children(
        monkeypatch, tmp_path, {"delay_ms": 50, "loss_pct": 0})
    assert mode == "in-process-shim"
    assert env["LORALINK_NET_SHIM"] == "50,0"


def test_working_tc_is_used_and_recorded(monkeypatch, tmp_path):
    monkeypatch.setattr(cluster_launch, "_tc_available", lambda: True)
    monkeypatch.setattr(cluster_launch, "_apply_netem", lambda parts: None)
    mode, env = _run_with_stubbed_children(
        monkeypatch, tmp_path, {"delay_ms": 25, "loss_pct": 0})
    assert mode == "tc-netem"
    assert "LORALINK_NET_SHIM" not in env, "shim must not double-apply over tc"


def test_shim_models_loss_as_retransmission_not_disconnection():
    """A lost segment costs an RTO; it does not kill the run.

    Modelling loss as ConnectionError measures LoraLink's missing retry path,
    which is a different question from how it behaves on a lossy link -- and it
    aborted every loss cell in the sweep.
    """
    import ast
    import inspect
    import pathlib

    src = pathlib.Path(inspect.getfile(cluster_launch)).parents[1] / "main.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    shim = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "_shim_send")

    raises = [n for n in ast.walk(shim) if isinstance(n, ast.Raise)]
    assert not raises, "shim must not raise on simulated loss"
    sleeps = [n for n in ast.walk(shim)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "sleep"]
    assert len(sleeps) >= 2, "expected a delay sleep and a retransmission sleep"
