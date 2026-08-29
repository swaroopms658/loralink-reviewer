import argparse
import subprocess
import sys
import types

import pytest


# --- existing CLI-surface tests (keep) -------------------------------------

def test_help_lists_new_flags(repo_root):
    out = subprocess.run([sys.executable, "main.py", "--help"],
                         cwd=repo_root, capture_output=True, text=True).stdout
    for flag in ["--seed", "--num-samples", "--epochs", "--partition-strategy",
                 "--run-tag", "--metrics-csv", "--base-model", "--eval-holdout"]:
        assert flag in out, flag


def test_bad_strategy_rejected(repo_root):
    r = subprocess.run([sys.executable, "main.py", "--role", "worker",
                        "--partition-strategy", "nope"],
                       cwd=repo_root, capture_output=True, text=True)
    assert r.returncode != 0 and "invalid choice" in r.stderr


# --- wiring tests: assert flags actually reach their targets --------------

def _import_main(repo_root):
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import main
    return main


def _fake_pipeline():
    ce = types.SimpleNamespace(
        get_compression_stats=lambda: {"average_compression_ratio": "2.00x"},
        stats={"total_compressed_bytes": 50, "total_original_bytes": 100},
    )
    return types.SimpleNamespace(
        compression_engine=ce,
        target_modules=["q_proj"],
        forward_step_local=lambda gb, batch: None,
        backward_step=lambda rid, grad: None,
        get_lora_state_dict=lambda: {},
        config=types.SimpleNamespace(master_ip="1.2.3.4", lora_rank=8),
    )


def _fake_device_manager_cls(partition_calls):
    class FakeDM:
        def __init__(self, ips, model_config):
            self.model_config = model_config
            self.devices = {
                ip: types.SimpleNamespace(status=types.SimpleNamespace(name="HEALTHY"))
                for ip in ips
            }

        def process_benchmark_result(self, ip, payload):
            pass

        def partition_model(self, master_ip, strategy="smart"):
            partition_calls.append(strategy)
            cfg = types.SimpleNamespace(
                assigned_layers=[0, 1], lora_rank=8, predecessor_ip=None,
                successor_ip=None, master_ip=master_ip, world_size=1,
                device_rank=0, model_name="m",
            )
            return {master_ip: cfg}

    return FakeDM


def _base_patches(main, monkeypatch, partition_calls, nm_bind):
    monkeypatch.setattr(
        main, "TeeLogger",
        lambda log_dir=".": types.SimpleNamespace(
            log_path="x", write=sys.stderr.write, flush=sys.stderr.flush),
    )

    def fake_nm(bind, port, handler):
        nm_bind.append(bind)
        return types.SimpleNamespace(
            start_server=lambda: None, stop_server=lambda: None,
            send_message=lambda *a, **k: None)

    monkeypatch.setattr(main, "NetworkManager", fake_nm)
    monkeypatch.setattr(main.benchmarking, "run_benchmark", lambda: {})
    monkeypatch.setattr(main, "DeviceManager", _fake_device_manager_cls(partition_calls))
    monkeypatch.setattr(main, "PipelineStage", lambda cfg, nm: _fake_pipeline())
    monkeypatch.setattr(
        main.AutoTokenizer, "from_pretrained",
        lambda *a, **k: types.SimpleNamespace(pad_token="p", eos_token="e"))


def _ns(**over):
    d = dict(role="coordinator", workers="10.0.0.2", host_ip="127.0.0.9",
             model_path="m", dataset="wikitext", seed=7, num_samples=11,
             epochs=1, eval_holdout=3, partition_strategy="round_robin",
             run_tag="t", metrics_csv="")
    d.update(over)
    return argparse.Namespace(**d)


def test_num_samples_and_strategy_and_bind_reach_targets(repo_root, monkeypatch):
    main = _import_main(repo_root)
    partition_calls, nm_bind = [], []
    _base_patches(main, monkeypatch, partition_calls, nm_bind)

    captured = {}

    class _Stop(Exception):
        pass

    def fake_loader(tok, **kw):
        captured.update(kw)
        raise _Stop

    monkeypatch.setattr(main.data_loader, "get_data_loader", fake_loader)

    ns = _ns()
    saved_stdout = sys.stdout
    try:
        with pytest.raises(_Stop):
            main.run_coordinator(ns)
    finally:
        sys.stdout = saved_stdout

    assert captured["num_samples"] == ns.num_samples
    assert captured["eval_holdout"] == ns.eval_holdout
    assert captured["split"] == "train"
    assert captured["dataset_name"] == ns.dataset
    assert partition_calls == [ns.partition_strategy]
    assert nm_bind[0] == ns.host_ip  # NetworkManager binds host_ip


def test_metrics_csv_written(repo_root, monkeypatch, tmp_path):
    main = _import_main(repo_root)
    partition_calls, nm_bind = [], []
    _base_patches(main, monkeypatch, partition_calls, nm_bind)

    monkeypatch.setattr(main.data_loader, "get_data_loader",
                        lambda tok, **kw: [{"b": 0}, {"b": 1}])
    monkeypatch.setattr(main, "save_lora_adapters", lambda **k: None)
    monkeypatch.setattr(main.ALL_WEIGHTS_RECEIVED, "wait", lambda timeout=None: True)

    import queue as _q
    fake_gq = _q.Queue()
    for i in range(4):  # epochs=2 * 2 batches
        fake_gq.put((i, None, 0.5))
    monkeypatch.setattr(main, "GRADIENT_QUEUE", fake_gq)

    csv_path = tmp_path / "m.csv"
    ns = _ns(epochs=2, metrics_csv=str(csv_path))

    saved_stdout = sys.stdout
    try:
        main.run_coordinator(ns)
    finally:
        sys.stdout = saved_stdout

    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    # header + 4 per-batch RUN rows + 1 SUMMARY row
    assert len(lines) == 6
    assert "global_batch" in lines[0]
    assert str(ns.seed) in lines[-1]
    assert ns.partition_strategy in lines[-1]
    assert "round_robin" in "".join(lines[1:])


def test_worker_binds_host_ip(repo_root, monkeypatch):
    main = _import_main(repo_root)
    nm_bind = []

    def fake_nm(bind, port, handler):
        nm_bind.append(bind)
        return types.SimpleNamespace(start_server=lambda: None,
                                     stop_server=lambda: None)

    monkeypatch.setattr(main, "NetworkManager", fake_nm)

    class _Woke(Exception):
        pass

    def _raise(*a):
        raise _Woke

    monkeypatch.setattr(main.time, "sleep", _raise)

    with pytest.raises(_Woke):
        main.run_worker(argparse.Namespace(host_ip="127.0.0.5", seed=1))

    assert nm_bind == ["127.0.0.5"]
