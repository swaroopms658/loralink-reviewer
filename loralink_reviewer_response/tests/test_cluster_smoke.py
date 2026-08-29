import inspect

import pytest

from loralink_reviewer_response import cluster_launch
from loralink_reviewer_response.cluster_launch import run_cluster


def test_ip_plan_and_signature():
    # REPO_ROOT must point at the repo root that holds main.py
    assert (cluster_launch.REPO_ROOT / "main.py").is_file()

    # IP plan: coordinator 127.0.0.1, workers 127.0.0.2 .. 127.0.0.(n+1)
    coord, workers = cluster_launch._ip_plan(3)
    assert coord == "127.0.0.1"
    assert workers == ["127.0.0.2", "127.0.0.3", "127.0.0.4"]

    # run_cluster signature must match exactly what the notebooks depend on
    sig = inspect.signature(run_cluster)
    params = sig.parameters
    assert list(params)[:3] == ["n_workers", "dataset", "seed"]
    for name in ("n_workers", "dataset", "seed"):
        assert params[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    expected_kw = {
        "model": "EleutherAI/gpt-neo-125M",
        "strategy": "smart",
        "compression": True,
        "num_samples": 60,
        "epochs": 1,
        "eval_holdout": 0,
        "netem": None,
        "tag": "",
        "run_timeout_s": 900,
        "results_csv": "results.csv",
        "save_adapters_to": None,
        "workdir": ".",
    }
    for name, default in expected_kw.items():
        assert params[name].kind is inspect.Parameter.KEYWORD_ONLY, name
        assert params[name].default == default, name


@pytest.mark.colab
def test_two_worker_run_writes_rows(tmp_path):
    import pandas as pd

    csv = run_cluster(
        n_workers=2, dataset="wikitext", seed=0,
        model="EleutherAI/gpt-neo-125M", num_samples=6, epochs=1,
        tag="smoke", results_csv=str(tmp_path / "r.csv"),
        run_timeout_s=600, workdir=".")
    df = pd.read_csv(csv)
    assert (df["run_tag"] == "smoke").any()
    assert df["loss"].dropna().shape[0] >= 3
    assert df["step_latency_s"].dropna().gt(0).all()
