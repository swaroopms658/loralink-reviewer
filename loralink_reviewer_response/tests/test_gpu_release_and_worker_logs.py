"""Two operational fixes found running NB02 (Phi-1.5) on a real T4.

1. `evaluate_adapter` loaded a full fp32 base model onto the GPU in the notebook
   process and never released it. NB02 evaluates between arms, so the ON arm's
   eval left ~5.3 GB of a 14.5 GB T4 held while the OFF arm tried to start three
   more model-loading processes.

2. `run_cluster` captured the coordinator's output but let worker output vanish,
   so a worker that died mid-run surfaced only as the coordinator's 300 s
   gradient timeout with no cause attached.
"""
import gc
import weakref

import pytest

from loralink_reviewer_response import cluster_launch, eval_quality


class _Dummy:
    def __init__(self):
        self.buf = [0] * 1024

    def to(self, *a, **k):
        return self


def test_release_model_drops_the_reference():
    model = _Dummy()
    ref = weakref.ref(model)
    eval_quality._release_model(model)
    del model
    gc.collect()
    assert ref() is None, "model must not stay reachable after release"


def test_release_model_survives_a_missing_cuda(monkeypatch):
    """Must not raise on a CPU-only box."""
    monkeypatch.setattr(eval_quality.torch.cuda, "is_available", lambda: False)
    eval_quality._release_model(_Dummy())          # must not raise


def test_release_model_tolerates_none():
    eval_quality._release_model(None)               # must not raise


def test_evaluate_adapter_releases_even_on_failure(monkeypatch, tmp_path):
    """A crash mid-eval must still free the GPU, or the next arm inherits it."""
    released = []
    monkeypatch.setattr(eval_quality, "_load",
                        lambda *a, **k: (_Dummy(), object()))
    monkeypatch.setattr(eval_quality, "_release_model",
                        lambda m: released.append(m))

    def _boom(*a, **k):
        raise RuntimeError("eval exploded")

    monkeypatch.setattr(eval_quality, "_load_e2e", _boom)

    with pytest.raises(RuntimeError, match="exploded"):
        eval_quality.evaluate_adapter("m", "a", "e2e",
                                      out_csv=str(tmp_path / "q.csv"))
    assert released, "release must run even when evaluation raises"


def test_release_does_not_copy_the_model_to_host_ram():
    """`.to("cpu")` would move ~5.6 GB into a 12.7 GB host at the worst moment."""
    import ast
    import inspect
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(eval_quality._release_model)))
    moves = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute) and node.func.attr == "to"
        and any(isinstance(a, ast.Constant) and a.value == "cpu" for a in node.args)
    ]
    assert not moves, "release must not stage the model through host RAM"


def test_eval_cli_exposes_the_expected_flags():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "loralink_reviewer_response.eval_quality", "--help"],
        capture_output=True, text=True).stdout
    for flag in ["--base-model", "--adapter-dir", "--dataset", "--arm",
                 "--seed", "--limit", "--out-csv"]:
        assert flag in out, flag


def test_subprocess_helper_builds_a_module_invocation(monkeypatch):
    seen = {}

    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    def _fake_run(cmd, **kw):
        seen["cmd"] = cmd
        seen["kw"] = kw
        return _R()

    monkeypatch.setattr(eval_quality.subprocess, "run", _fake_run)
    eval_quality.evaluate_adapter_subprocess(
        "microsoft/phi-1_5", "adapters/x", "e2e", arm="ON", seed=2,
        limit=200, out_csv="q.csv")

    cmd = seen["cmd"]
    assert "-m" in cmd and "loralink_reviewer_response.eval_quality" in cmd
    assert "--adapter-dir" in cmd and "adapters/x" in cmd
    assert cmd[cmd.index("--arm") + 1] == "ON"
    assert cmd[cmd.index("--seed") + 1] == "2"
    assert seen["kw"].get("timeout")


def test_subprocess_helper_omits_adapter_dir_when_absent(monkeypatch):
    class _R:
        returncode = 0
        stdout = ""
        stderr = ""

    seen = {}
    monkeypatch.setattr(eval_quality.subprocess, "run",
                        lambda cmd, **kw: (seen.update(cmd=cmd), _R())[1])
    eval_quality.evaluate_adapter_subprocess("m", None, "wikitext")
    assert "--adapter-dir" not in seen["cmd"], "base-only eval takes no adapter"


def test_subprocess_helper_raises_with_child_output(monkeypatch):
    class _R:
        returncode = 1
        stdout = "some stdout"
        stderr = "CUDA out of memory"

    monkeypatch.setattr(eval_quality.subprocess, "run", lambda cmd, **kw: _R())
    with pytest.raises(RuntimeError, match="out of memory"):
        eval_quality.evaluate_adapter_subprocess("m", None, "e2e")


def test_child_processes_run_unbuffered():
    """Buffered child stdout is discarded on SIGTERM, hiding the failure cause."""
    wcmd = cluster_launch._worker_cmd("127.0.0.2", "m", 0)
    ccmd = cluster_launch._coord_cmd("127.0.0.1", ["127.0.0.2"], model="m",
                                     dataset="wikitext", seed=0, num_samples=1,
                                     epochs=1, eval_holdout=0, strategy="smart",
                                     tag="t", csv_path="x.csv")
    assert "-u" in wcmd, "worker must run unbuffered"
    assert "-u" in ccmd, "coordinator must run unbuffered"
    assert wcmd.index("-u") < wcmd.index("--role"), "-u must precede the script"


def test_worker_errors_print_a_traceback():
    """A bare str(e) on a network thread is not enough to diagnose a stall."""
    import inspect
    import re

    import main

    src = inspect.getsource(main._worker_message_handler)
    for label in ("Error processing tensor", "Error processing gradient"):
        idx = src.index(label)
        window = src[idx:idx + 300]
        assert re.search(r"traceback\.print_exc\(\)", window), label


def test_worker_log_paths_are_per_worker():
    paths = cluster_launch._worker_log_paths(tmp := __import__("pathlib").Path("/tmp/x.csv"), 3)
    assert len(paths) == 3
    assert len(set(paths)) == 3, "each worker needs its own log file"
    for p in paths:
        assert str(p).startswith(str(tmp.parent))


def test_worker_log_tail_reports_missing_and_empty(tmp_path):
    present = tmp_path / "w0.log"
    present.write_text("line one\nline two\n", encoding="utf-8")
    empty = tmp_path / "w1.log"
    empty.write_text("", encoding="utf-8")
    missing = tmp_path / "w2.log"

    out = cluster_launch._worker_log_tail([present, empty, missing], limit=5)
    assert "line two" in out
    assert "w1" in out and "w2" in out, "every worker should be accounted for"


def test_worker_log_tail_is_bounded(tmp_path):
    p = tmp_path / "w0.log"
    p.write_text("\n".join(f"line {i}" for i in range(500)), encoding="utf-8")
    out = cluster_launch._worker_log_tail([p], limit=5)
    assert "line 499" in out
    assert "line 400" not in out, "tail must be bounded"
