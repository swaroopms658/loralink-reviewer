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
