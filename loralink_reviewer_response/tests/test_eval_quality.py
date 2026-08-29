"""Tests for eval_quality (Task 9).

Two offline tests (no mark) + one colab test that actually runs a model.
"""
import ast
import inspect
import pathlib
import sys

import pytest

# eval_quality lives in the package dir; make the bare import work regardless of
# how pytest is invoked.
_PKG = pathlib.Path(__file__).resolve().parents[1]
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))

import eval_quality  # noqa: E402
from eval_quality import evaluate_adapter  # noqa: E402


def test_signature_and_cols():
    sig = inspect.signature(evaluate_adapter)
    params = list(sig.parameters.values())
    names = [p.name for p in params]
    assert names == ["base_model", "adapter_dir", "dataset", "eval_holdout",
                     "max_new_tokens", "limit", "arm", "seed", "out_csv"]

    kinds = {p.name: p.kind for p in params}
    for pos in ("base_model", "adapter_dir", "dataset"):
        assert kinds[pos] == inspect.Parameter.POSITIONAL_OR_KEYWORD
    for kw in ("eval_holdout", "max_new_tokens", "limit", "arm", "seed", "out_csv"):
        assert kinds[kw] == inspect.Parameter.KEYWORD_ONLY

    defaults = {p.name: p.default for p in params if p.default is not inspect._empty}
    assert defaults == {"eval_holdout": 200, "max_new_tokens": 48, "limit": 100,
                        "arm": "", "seed": 0, "out_csv": "results_quality.csv"}

    assert eval_quality._COLS == ["arm", "seed", "dataset", "base_model",
                                  "perplexity", "bleu", "rougeL", "n_eval",
                                  "adapter_dir", "slice_bounds"]


def test_module_imports_without_peft_or_evaluate():
    # It already imported above without raising.
    leaked = {m for m in ("peft", "evaluate") if m in sys.modules}
    if not leaked:
        return
    # Something (e.g. transformers) pulled them in transitively -> prove the
    # lazy-import contract via source inspection instead.
    tree = ast.parse(pathlib.Path(eval_quality.__file__).read_text(encoding="utf-8"))
    top_level = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level += [n.name.split(".")[0] for n in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level.append(node.module.split(".")[0])
    assert "peft" not in top_level
    assert "evaluate" not in top_level


@pytest.mark.colab
def test_eval_runs_on_base_only(tmp_path):
    out_csv = tmp_path / "q.csv"
    r = evaluate_adapter("EleutherAI/gpt-neo-125M", "", "wikitext",
                         eval_holdout=20, limit=10, out_csv=str(out_csv))
    assert r["perplexity"] > 1.0
    assert r["n_eval"] == 10
    assert r["bleu"] == "" and r["rougeL"] == ""
    assert r["adapter_dir"] == "(base)"

    lines = out_csv.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",")[:4] == ["arm", "seed", "dataset", "base_model"]
    assert len(lines) == 2
