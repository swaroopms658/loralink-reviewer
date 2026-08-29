import os
import torch
import importlib

def _fresh_engine(monkeypatch, value):
    monkeypatch.setenv("LORALINK_LOSSY_COMPRESSION", value)
    import compression_engine
    importlib.reload(compression_engine)
    return compression_engine.OptimizedCompressionEngine()

def test_toggle_off_is_lossless(monkeypatch):
    eng = _fresh_engine(monkeypatch, "0")
    for p in eng.compression_params.values():
        assert p["sparsity_ratio"] == 0.0 and p["quantize"] is False
    x = torch.randn(64, 32, dtype=torch.float32)
    back = eng.decompress_tensor(eng.compress_tensor(x, "gradients"))
    assert torch.equal(back, x)

def test_toggle_on_is_default_lossy(monkeypatch):
    eng = _fresh_engine(monkeypatch, "1")
    assert eng.compression_params["gradients"]["quantize"] is True
    assert eng.compression_params["gradients"]["sparsity_ratio"] == 0.7

def test_fake_benchmark(monkeypatch):
    monkeypatch.setenv("LORALINK_FAKE_BENCHMARK", "1")
    import benchmarking, time
    importlib.reload(benchmarking)
    t0 = time.perf_counter()
    r = benchmarking.run_benchmark()
    assert time.perf_counter() - t0 < 0.5
    assert r["flops"] == 5.0 and r["memory_gb"] > 0
