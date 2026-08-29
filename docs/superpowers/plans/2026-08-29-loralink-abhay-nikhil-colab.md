# LoraLink Abhay & Nikhil Reviewer Response — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Colab-Free-Tier-runnable package that answers the eight Abhay & Nikhil reviewer concerns for the LoraLink paper — benchmarking *only our implementation*, using published numbers for every competitor method.

**Architecture:** A small additive patch to the existing LoraLink repo (new CLI flags, a compression on/off env toggle, extra partitioners, a metrics CSV writer, an eval-data split) plus a set of standalone orchestration/eval modules (`cluster_launch.py`, `eval_quality.py`, `aggregate.py`) and seven sharded Colab notebooks. Each notebook runs one experiment in ≤ 40 min, dumps a CSV, and the user downloads it; a final notebook merges all CSVs with a curated published-baselines table into comparison tables, plots, and a reviewer-response document. No LoraLink algorithm, protocol, or LoRA-reconstruction code changes.

**Tech Stack:** Python 3, PyTorch (CPU locally / CUDA T4 on Colab), HuggingFace `transformers` / `datasets` / `peft` / `evaluate`, `zstandard`, `scipy`, `pandas`, `matplotlib`, Linux `tc`/`netem`, Jupyter/Colab.

**Spec:** `docs/superpowers/specs/2026-08-29-loralink-abhay-nikhil-colab-design.md` (read it alongside this plan — the plan argues from it)

## Global Constraints

- **Benchmark our implementation only.** Never run, re-implement, or approximate DeepSpeed, FSDP, SplitLoRA, HSplitLoRA, Petals, QLoRA, or Megatron-LM. Their numbers are transcribed from published papers into `baselines/published_baselines.csv` with a verbatim source quote in `baselines/SOURCES.md`.
- **No optics tuning.** The patch may only add: CLI flags, bind-address selection, a lossless-only compression toggle, extra partitioner strategies, metric logging, an eval-data split, and a benchmark fast-path. Nothing touches the optimizer, LoRA math, gradient/activation compression math, wire format, or reconstruction.
- **Model tiers (ours):** `EleutherAI/gpt-neo-125M` for NB 01/03/04/05; `microsoft/phi-1_5` (1.3 B) for NB 02/02b. Nothing larger.
- **Datasets (ours):** `wikitext/wikitext-2-raw-v1` (perplexity), `GEM/e2e_nlg` (BLEU + ROUGE-L). No Dolly in our runs.
- **Seeds:** NB 01 `{0,1,2,3,4}`; NB 02 `{0,1,2}`; NB 03 `{0,1,2}`; NB 04/05 `{0}` with 3 internal timing repeats. Confidence intervals: Student-t, state `n`, flag as small.
- **Training length:** NB 01 = 60 mini-batches / 1 epoch; NB 02 = 50 / 1 epoch; NB 02b = 3 epochs (~150 batches); NB 03/04/05 = 20–30 batches (timing only).
- **Per-notebook wall ceiling: 40 min on Free T4**, including install + model download. Every notebook has a `WALL_BUDGET_MIN = 32` guard that stops the run loop and saves a partial CSV; `run_cluster` kills any single run exceeding `run_timeout_s` (default 900 s).
- **Provenance tags:** every reported number is `[ours]` or `[published, ref N]`. Loopback runs (NB 04/05) are labelled "single-box simulation — loopback + emulation, not WAN" in every caption, table, and CSV (`sim` column).
- **Deliverable root:** `loralink_reviewer_response/` at the repo top level. The patch is applied to a *copy* of the repo inside each Colab session, never committed over the originals.
- **Repo layout:** the existing LoraLink source files (`main.py`, `device_manager.py`, `compression_engine.py`, `data_loader.py`, `pipeline_engine.py`, `network_protocol.py`, `benchmarking.py`, `model_registry.py`, `lora_manager.py`, `downloader.py`) live at the repo root. This directory is **not** a git repo (spec §9).

### Git note

The repo is not under git. Task 0 offers `git init` for the deliverable work only. If accepted, run the `Commit` step at the end of each task as written. If declined, replace every `Commit` step with: "Save all files; run `python loralink_reviewer_response/patch/checksums.py --update` if repo source changed; tick the task box." The plan is written assuming git is initialised.

### Local vs Colab tests

- **[local]** — runs on this Windows box, CPU-only, offline. Pure-Python logic: `metrics_logger`, `compute_assignments`, compression toggle, stat helpers, CSV aggregation.
- **[colab]** — needs GPU / network / a live cluster / large deps. Cluster smoke, `eval_quality`, full notebooks. These get a runnable command and expected output in the plan; the executor runs them in a Colab session or a local session with deps installed and marks the box when verified.

---

## File Structure

### Patched repo source (modified in place, additively)

| File | Responsibility after patch |
|---|---|
| `main.py` | + argparse flags (`--seed --num-samples --epochs --partition-strategy --run-tag --metrics-csv --base-model`), seed both roles, bind `NetworkManager` to `--host-ip`, epoch loop, per-batch + summary metrics rows. |
| `compression_engine.py` | + `LORALINK_LOSSY_COMPRESSION=0` env → sparsity 0 / no quant (lossless zstd only). |
| `device_manager.py` | + `compute_assignments(strategy, devices, num_layers, layer_size_gb, embedding_size_gb, master_ip, seed)` pure function; `partition_model(master_ip, strategy="smart")` dispatches; strategies `smart` (existing), `round_robin`, `proportional`, `random`. |
| `data_loader.py` | + `get_data_loader(..., split="train", eval_holdout=0)` — disjoint held-out slice for evaluation, logged bounds. |
| `benchmarking.py` | + `LORALINK_FAKE_BENCHMARK=1` env → return synthetic stats instantly (scalability notebook only). |

### New — deliverable package

| File | Responsibility |
|---|---|
| `loralink_reviewer_response/metrics_logger.py` | `append_rows(csv_path, rows)` + `RUN_COLUMNS`, `SUMMARY_COLUMNS`. Header-once CSV append. |
| `loralink_reviewer_response/cluster_launch.py` | `run_cluster(...)` — spawn 1 coordinator + N workers on `127.0.0.x`, optional `tc netem` / in-process net shim, per-run timeout, teardown, returns metrics CSV path. |
| `loralink_reviewer_response/eval_quality.py` | Load base + `./lora_adapters` via `peft`, compute held-out perplexity + BLEU + ROUGE-L, append a row per adapter. |
| `loralink_reviewer_response/aggregate.py` | Merge `results/*.csv` + `baselines/published_baselines.csv` → tables (`figures/T*.csv`) + plots (`figures/*.png/.pdf`) + fill `RESPONSE_ABHAY_NIKHIL.md`. |
| `loralink_reviewer_response/statlib.py` | `mean_std_ci(values, confidence=0.95)` → `(mean, std, lo, hi, n)`; shared by aggregate + notebooks. |
| `loralink_reviewer_response/patch/apply_patch.py` | Apply the five source diffs to a repo copy; verify with `checksums.py`. |
| `loralink_reviewer_response/patch/*.patch` | Unified diffs for the five files above. |
| `loralink_reviewer_response/patch/checksums.py` | Record / verify SHA-256 of the five patched source files (provenance, since no git commit hash). |
| `loralink_reviewer_response/baselines/published_baselines.csv` | One row per cited published number. |
| `loralink_reviewer_response/baselines/SOURCES.md` | Per source: citation, URL/DOI, table/section, verbatim quoted value. |
| `loralink_reviewer_response/notebooks/_template.ipynb` | Canonical 5-cell notebook skeleton the others are generated from. |
| `loralink_reviewer_response/notebooks/00_setup_smoke.ipynb` | Deps + patch + 125M download + a 10-batch 2-worker run; asserts a CSV row was written. |
| `loralink_reviewer_response/notebooks/01_stat_validation.ipynb` | 125M, one dataset per shard, 5 seeds × 60 batches, compression ON → `results_stat_<tag>.csv`. |
| `loralink_reviewer_response/notebooks/02_task_quality.ipynb` | Phi-1.5, one `(dataset, seed)` per shard, arms {ON, OFF, reference}, 50 batches, saves adapters, calls `eval_quality` → `results_quality_<tag>.csv`. |
| `loralink_reviewer_response/notebooks/02b_convergence.ipynb` | Phi-1.5, 1 seed, arm ON, 1 dataset, 3 epochs, per-batch loss → `results_converge_<tag>.csv`. Run once. |
| `loralink_reviewer_response/notebooks/03_alt_scheduling.ipynb` | 125M, strategies {smart, round_robin, proportional, random} × 3 seeds × 30 batches → `results_sched_<tag>.csv`. |
| `loralink_reviewer_response/notebooks/04_scalability_sim.ipynb` | 125M, N ∈ {2,3,4,5,6,8} workers × 30 batches × 3 repeats, `FAKE_BENCHMARK` → `results_scale_<tag>.csv`. |
| `loralink_reviewer_response/notebooks/05_network_netem.ipynb` | 125M, delay {0,25,50,100 ms} × loss {0,1,3 %} × rate {none,10,50 mbit} subset, 20 batches → `results_net_<tag>.csv`. |
| `loralink_reviewer_response/notebooks/99_aggregate_report.ipynb` | Runs `aggregate.py` on `results/`; renders tables + plots inline; writes `RESPONSE_ABHAY_NIKHIL.md`. |
| `loralink_reviewer_response/RESPONSE_ABHAY_NIKHIL.md` | Reviewer-response prose, numbers slotted by aggregate, every claim tagged. |
| `loralink_reviewer_response/README.md` | Datasets, protocol, metrics, per-number provenance, how to run each notebook/shard. |
| `loralink_reviewer_response/tests/` | pytest: `test_metrics_logger.py`, `test_compute_assignments.py`, `test_compression_toggle.py`, `test_statlib.py`, `test_data_holdout.py`, `test_aggregate.py`, `test_cluster_smoke.py`. |

---

## Task 0: Scaffold + git decision

**Files:**
- Create: `loralink_reviewer_response/` tree (empty dirs + `__init__`-free), `loralink_reviewer_response/tests/conftest.py`
- Create: `loralink_reviewer_response/requirements-colab.txt`
- Create: `.gitignore` (if git accepted)

**Interfaces:**
- Consumes: nothing
- Produces: directory layout; `REPO_ROOT` fixture (`conftest.py`) returning the repo top-level `Path`; `requirements-colab.txt` pinning `peft`, `evaluate`, `rouge-score`, `sacrebleu`, `scipy`, `pandas`, `matplotlib`, `zstandard`, `psutil`, `datasets`, `transformers`, `accelerate`.

- [ ] **Step 1: Ask the user the git question**

Ask: "The repo isn't under git. Want me to `git init` so the deliverable work is tracked and the patch is diffable? (y/n)". If yes, `git init && git add -A && git commit -m "chore: baseline before reviewer-response work"`. If no, note it and use the checksum-based provenance path.

- [ ] **Step 2: Create the directory tree**

```bash
mkdir -p loralink_reviewer_response/{notebooks,patch,baselines,results,figures,tests}
```

- [ ] **Step 3: Write `requirements-colab.txt`**

```
transformers==4.44.2
datasets==2.21.0
accelerate==0.34.2
peft==0.12.0
evaluate==0.4.2
rouge-score==0.1.2
sacrebleu==2.4.3
scipy==1.13.1
pandas==2.2.2
matplotlib==3.9.2
zstandard==0.23.0
psutil==6.0.0
```
(Executor: bump to the newest compatible pins if Colab's preinstalled `torch` forces it; record final pins in `README.md`.)

- [ ] **Step 4: Write `tests/conftest.py`**

```python
import pathlib
import pytest

@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    # tests/ -> loralink_reviewer_response/ -> repo root
    return pathlib.Path(__file__).resolve().parents[2]

@pytest.fixture(scope="session")
def pkg_dir(repo_root) -> pathlib.Path:
    return repo_root / "loralink_reviewer_response"
```

- [ ] **Step 5: Commit**

```bash
git add loralink_reviewer_response .gitignore
git commit -m "chore: scaffold reviewer-response package"
```

---

## Task 1: `metrics_logger.py`

**Files:**
- Create: `loralink_reviewer_response/metrics_logger.py`
- Test: `loralink_reviewer_response/tests/test_metrics_logger.py`

**Interfaces:**
- Consumes: nothing
- Produces:
  - `RUN_COLUMNS: list[str]` = `["run_tag","seed","strategy","compression","dataset","model","n_workers","epoch","global_batch","loss","step_latency_s","comp_ratio","bytes_sent","bytes_saved","sim","timestamp"]`
  - `SUMMARY_COLUMNS: list[str]` = `["run_tag","seed","strategy","compression","dataset","model","n_workers","sim","n_batches","mean_loss","last_loss","mean_step_latency_s","total_bytes_sent","total_bytes_saved","overall_comp_ratio","wall_time_s","partition_map","partition_balance_std"]`
  - `append_rows(csv_path: str | Path, rows: list[dict], columns: list[str]) -> None` — creates parent dir, writes header iff file absent/empty, appends rows, fills missing keys with `""`, ignores extra keys.

- [ ] **Step 1: Write the failing test** — `tests/test_metrics_logger.py`

```python
import csv
from loralink_reviewer_response.metrics_logger import append_rows, RUN_COLUMNS

def test_header_written_once_and_rows_appended(tmp_path):
    p = tmp_path / "m.csv"
    append_rows(p, [{"run_tag": "a", "seed": 0, "loss": 1.5}], RUN_COLUMNS)
    append_rows(p, [{"run_tag": "a", "seed": 1, "loss": 1.2}], RUN_COLUMNS)
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == RUN_COLUMNS            # header once
    assert rows[1][RUN_COLUMNS.index("loss")] == "1.5"
    assert rows[2][RUN_COLUMNS.index("seed")] == "1"
    assert len(rows) == 3

def test_missing_keys_become_blank_and_extra_keys_ignored(tmp_path):
    p = tmp_path / "m.csv"
    append_rows(p, [{"run_tag": "x", "nonsense": 9}], RUN_COLUMNS)
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[1][RUN_COLUMNS.index("run_tag")] == "x"
    assert rows[1][RUN_COLUMNS.index("loss")] == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest loralink_reviewer_response/tests/test_metrics_logger.py -v`
Expected: FAIL — `ModuleNotFoundError: loralink_reviewer_response.metrics_logger`

- [ ] **Step 3: Implement `metrics_logger.py`**

```python
"""Append-only CSV writer for LoraLink reviewer-response experiments."""
from __future__ import annotations
import csv
import os
from pathlib import Path

RUN_COLUMNS = [
    "run_tag", "seed", "strategy", "compression", "dataset", "model",
    "n_workers", "epoch", "global_batch", "loss", "step_latency_s",
    "comp_ratio", "bytes_sent", "bytes_saved", "sim", "timestamp",
]
SUMMARY_COLUMNS = [
    "run_tag", "seed", "strategy", "compression", "dataset", "model",
    "n_workers", "sim", "n_batches", "mean_loss", "last_loss",
    "mean_step_latency_s", "total_bytes_sent", "total_bytes_saved",
    "overall_comp_ratio", "wall_time_s", "partition_map",
    "partition_balance_std",
]

def append_rows(csv_path, rows, columns):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if need_header:
            w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest loralink_reviewer_response/tests/test_metrics_logger.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add loralink_reviewer_response/metrics_logger.py loralink_reviewer_response/tests/test_metrics_logger.py
git commit -m "feat: metrics CSV logger"
```

---

## Task 2: `statlib.py`

**Files:**
- Create: `loralink_reviewer_response/statlib.py`
- Test: `loralink_reviewer_response/tests/test_statlib.py`

**Interfaces:**
- Consumes: `scipy.stats`
- Produces: `mean_std_ci(values: Sequence[float], confidence: float = 0.95) -> dict` with keys `mean, std, ci_lo, ci_hi, n`. `n < 2` → `ci_lo = ci_hi = mean`, `std = 0.0`. `std` is the sample std (ddof=1) for `n >= 2`.

- [ ] **Step 1: Write the failing test** — `tests/test_statlib.py`

```python
import math
from loralink_reviewer_response.statlib import mean_std_ci

def test_single_value_has_degenerate_ci():
    r = mean_std_ci([2.0])
    assert r["n"] == 1 and r["mean"] == 2.0
    assert r["ci_lo"] == 2.0 and r["ci_hi"] == 2.0 and r["std"] == 0.0

def test_known_sample():
    r = mean_std_ci([1.0, 2.0, 3.0, 4.0, 5.0])
    assert math.isclose(r["mean"], 3.0)
    assert math.isclose(r["std"], math.sqrt(2.5))          # ddof=1
    assert r["ci_lo"] < 3.0 < r["ci_hi"]
    assert math.isclose(r["ci_hi"] - 3.0, 3.0 - r["ci_lo"], rel_tol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest loralink_reviewer_response/tests/test_statlib.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `statlib.py`**

```python
"""Small-sample statistics for reviewer-response tables."""
from __future__ import annotations
from statistics import fmean, stdev
from scipy.stats import t as _t

def mean_std_ci(values, confidence: float = 0.95) -> dict:
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return {"mean": float("nan"), "std": 0.0, "ci_lo": float("nan"),
                "ci_hi": float("nan"), "n": 0}
    m = fmean(vals)
    if n < 2:
        return {"mean": m, "std": 0.0, "ci_lo": m, "ci_hi": m, "n": n}
    s = stdev(vals)                      # ddof=1
    half = _t.ppf(0.5 + confidence / 2, df=n - 1) * s / (n ** 0.5)
    return {"mean": m, "std": s, "ci_lo": m - half, "ci_hi": m + half, "n": n}
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest loralink_reviewer_response/tests/test_statlib.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add loralink_reviewer_response/statlib.py loralink_reviewer_response/tests/test_statlib.py
git commit -m "feat: small-sample CI helper"
```

---

## Task 3: `device_manager.py` — pure `compute_assignments` + strategies

**Files:**
- Modify: `device_manager.py` (add `compute_assignments`, refactor tail of `partition_model` to call it, add `strategy` param)
- Test: `loralink_reviewer_response/tests/test_compute_assignments.py`

**Interfaces:**
- Consumes: existing `DeviceStats`, `DeviceHandle`, `PipelineConfig`
- Produces:
  - `compute_assignments(strategy: str, devices: list[DeviceHandle], num_layers: int, layer_size_gb: float, embedding_size_gb: float, master_ip: str, utilization_limit: float, seed: int = 0) -> dict[str, int]` — pure; returns `{ip: layer_count}`, sums to `num_layers`, every returned device (count > 0) participates, coordinator (`master_ip`) capped at 1 for every strategy, raises `PartitionInfeasible` if a memory-aware strategy cannot place all layers.
  - `class PartitionInfeasible(RuntimeError)`
  - `DeviceManager.partition_model(self, master_ip, strategy="smart")` — unchanged return type (`dict[str, PipelineConfig]`); `strategy` threaded from caller.
- Strategy semantics:
  - `smart` — existing memory-aware body, refactored to emit `assignments` then reuse the shared config-builder tail.
  - `round_robin` — deal layers cyclically over `[coordinator] + workers` in cluster order; coordinator gets at most 1 (skip it once it has 1).
  - `proportional` — worker layer counts ∝ `device.stats.flops`, largest-remainder rounding; coordinator = 1; every active worker ≥ 1.
  - `random` — `random.Random(seed)` shuffles a bag of layer indices across workers with each worker ≥ 1; coordinator = 1.
  - Non-smart strategies still run the existing embedding/LM-head feasibility check; on failure they raise `PartitionInfeasible` (no silent repair — the comparison point is how naive schedulers break).

- [ ] **Step 1: Write the failing test** — `tests/test_compute_assignments.py`

```python
import pytest
from device_manager import DeviceHandle, DeviceStats, DeviceStatus, compute_assignments, PartitionInfeasible

def _dev(ip, flops, mem, dtype="cuda"):
    return DeviceHandle(ip=ip, status=DeviceStatus.HEALTHY,
                        stats=DeviceStats(flops=flops, memory_gb=mem, device_type=dtype))

def _cluster():
    return [
        _dev("127.0.0.1", 0.2, 12.0, "cpu"),   # coordinator
        _dev("127.0.0.2", 8.0, 15.0),
        _dev("127.0.0.3", 8.0, 15.0),
        _dev("127.0.0.4", 8.0, 15.0),
    ]

@pytest.mark.parametrize("strategy", ["smart", "round_robin", "proportional", "random"])
def test_all_layers_assigned_and_coordinator_capped(strategy):
    devs = _cluster()
    a = compute_assignments(strategy, devs, num_layers=12, layer_size_gb=0.02,
                            embedding_size_gb=0.1, master_ip="127.0.0.1",
                            utilization_limit=0.70, seed=0)
    assert sum(a.values()) == 12
    assert a["127.0.0.1"] <= 1
    assert all(c >= 0 for c in a.values())
    assert all(a[d.ip] >= 1 for d in devs[1:])          # every worker participates

def test_proportional_favours_faster_worker():
    devs = _cluster()
    devs[1].stats.flops = 40.0                           # 127.0.0.2 much faster
    a = compute_assignments("proportional", devs, 12, 0.02, 0.1,
                            "127.0.0.1", 0.70, seed=0)
    assert a["127.0.0.2"] > a["127.0.0.3"]

def test_infeasible_raises():
    devs = _cluster()
    for d in devs:
        d.stats.memory_gb = 0.05                         # nothing fits
    with pytest.raises(PartitionInfeasible):
        compute_assignments("round_robin", devs, 12, 0.5, 0.1,
                            "127.0.0.1", 0.70, seed=0)

def test_random_is_seed_deterministic():
    devs = _cluster()
    a1 = compute_assignments("random", devs, 12, 0.02, 0.1, "127.0.0.1", 0.70, seed=7)
    a2 = compute_assignments("random", devs, 12, 0.02, 0.1, "127.0.0.1", 0.70, seed=7)
    assert a1 == a2
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest loralink_reviewer_response/tests/test_compute_assignments.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_assignments'`

- [ ] **Step 3: Add `PartitionInfeasible` + `compute_assignments` to `device_manager.py`**

Insert after the `PipelineConfig` dataclass (≈ line 39):

```python
class PartitionInfeasible(RuntimeError):
    """A partition strategy could not place every layer within memory limits."""


def _usable_mem_gb(dev, utilization_limit, is_master, embedding_size_gb):
    if dev.stats.device_type == "cuda":
        u = dev.stats.memory_gb * utilization_limit
    else:
        # match partition_model: coordinator (RAM-4)*0.65, worker (RAM-5)*0.60
        u = max(0.0, dev.stats.memory_gb - (4.0 if is_master else 5.0)) * (0.65 if is_master else 0.60)
    if is_master:
        u = max(0.0, u - embedding_size_gb)
    return u


def _assert_feasible(assignments, devices, layer_size_gb, embedding_size_gb,
                     master_ip, utilization_limit):
    for d in devices:
        need = assignments.get(d.ip, 0) * layer_size_gb
        have = _usable_mem_gb(d, utilization_limit, d.ip == master_ip, embedding_size_gb)
        if need > have + 1e-9:
            raise PartitionInfeasible(
                f"{d.ip}: needs {need:.3f} GB for {assignments[d.ip]} layers, "
                f"has {have:.3f} GB usable")


def compute_assignments(strategy, devices, num_layers, layer_size_gb,
                        embedding_size_gb, master_ip, utilization_limit, seed=0):
    healthy = [d for d in devices if d.status == DeviceStatus.HEALTHY]
    assert healthy, "no healthy devices"
    ordered = sorted(healthy, key=lambda d: (0 if d.ip == master_ip else 1, -d.stats.flops))
    workers = [d for d in ordered if d.ip != master_ip]
    assert workers, "need at least one worker"

    if strategy == "smart":
        # delegate to the existing memory-aware routine, which is refactored
        # to expose its assignments dict (see Step 4).
        return _smart_assignments(ordered, num_layers, layer_size_gb,
                                  embedding_size_gb, master_ip, utilization_limit)

    a = {d.ip: 0 for d in ordered}
    coord_layer = 1 if num_layers > len(workers) else 0
    a[master_ip] = coord_layer
    rem = num_layers - coord_layer

    if strategy == "round_robin":
        i = 0
        while rem > 0:
            a[workers[i % len(workers)].ip] += 1
            rem -= 1
            i += 1
    elif strategy == "proportional":
        weights = [max(d.stats.flops, 1e-6) for d in workers]
        total = sum(weights)
        raw = [rem * w / total for w in weights]
        base = [int(x) for x in raw]
        for d, b in zip(workers, base):
            a[d.ip] = b
        leftover = rem - sum(base)
        for d, _ in sorted(zip(workers, raw), key=lambda p: p[1] - int(p[1]),
                           reverse=True)[:leftover]:
            a[d.ip] += 1
        for d in workers:                       # guarantee >= 1 by stealing from richest
            if a[d.ip] == 0:
                donor = max(workers, key=lambda x: a[x.ip])
                a[donor.ip] -= 1
                a[d.ip] += 1
    elif strategy == "random":
        rng = __import__("random").Random(seed)
        for d in workers:                       # floor of 1 each
            a[d.ip] += 1
        rem -= len(workers)
        for _ in range(max(0, rem)):
            a[rng.choice(workers).ip] += 1
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    assert sum(a.values()) == num_layers, (a, num_layers)
    _assert_feasible(a, ordered, layer_size_gb, embedding_size_gb,
                     master_ip, utilization_limit)
    return a
```

- [ ] **Step 4: Refactor `partition_model` to expose `_smart_assignments` and accept `strategy`**

- Extract the body of the current `partition_model` that computes the `assignments` dict (sections 2–6, ending before "# 7. Generate Configs") into a module-level `_smart_assignments(sorted_devices, num_layers, layer_size_gb, embedding_size_gb, master_ip, utilization_limit) -> dict[str, int]`. Keep its `sys.exit(1)` calls **only** for the coordinator-too-weak case; convert the "INSUFFICIENT MEMORY" exit into `raise PartitionInfeasible(...)` so all strategies share one failure type. (Smart's existing overflow handling stays — it is part of "smart".)
- Change the signature to `def partition_model(self, master_ip: str, strategy: str = "smart") -> dict[str, PipelineConfig]:`.
- Replace the inline assignment computation with:
  ```python
  assignments = compute_assignments(
      strategy, list(self.devices.values()), num_layers, layer_size_gb,
      embedding_size_gb, master_ip, self.utilization_limit,
      seed=self.model_config.get("seed", 0))
  sorted_devices = sorted(
      [d for d in self.devices.values() if d.status == DeviceStatus.HEALTHY],
      key=lambda d: (0 if d.ip == master_ip else 1, -d.stats.flops))
  ```
- Keep "# 7. Generate Configs and Filter Idle Devices" through `return configs` exactly as-is (it already reads `assignments[...]`).

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest loralink_reviewer_response/tests/test_compute_assignments.py -v`
Expected: PASS (5 passed). If `smart` path needs `AutoConfig` (network), mark that one param case `@pytest.mark.colab` and skip locally; the four non-smart cases must pass offline.

- [ ] **Step 6: Regression check — smart path unchanged**

Run: `python -m pytest loralink_reviewer_response/tests/test_compute_assignments.py -v -k smart` **[colab]** (needs `AutoConfig.from_pretrained` network access). Expected: PASS, and the printed layer map for the 4-device cluster matches what `main.py` printed before the patch (eyeball against a pre-patch run log if available).

- [ ] **Step 7: Commit**

```bash
git add device_manager.py loralink_reviewer_response/tests/test_compute_assignments.py
git commit -m "feat: alternative partition strategies (round_robin, proportional, random)"
```

---

## Task 4: `compression_engine.py` + `benchmarking.py` env toggles

**Files:**
- Modify: `compression_engine.py` (`OptimizedCompressionEngine.__init__`)
- Modify: `benchmarking.py` (`run_benchmark` fast-path)
- Test: `loralink_reviewer_response/tests/test_compression_toggle.py`

**Interfaces:**
- Consumes: env vars `LORALINK_LOSSY_COMPRESSION` (default `"1"`), `LORALINK_FAKE_BENCHMARK` (default `"0"`)
- Produces:
  - When `LORALINK_LOSSY_COMPRESSION=0`: `OptimizedCompressionEngine().compression_params` has every entry `{"sparsity_ratio": 0.0, "quantize": False}`; `compress_tensor` / `decompress_tensor` round-trip is **bit-exact** for float32 input (lossless).
  - When `LORALINK_FAKE_BENCHMARK=1`: `benchmarking.run_benchmark()` returns `{"flops": 5.0, "memory_gb": <detected>, "device_type": <detected>}` in < 50 ms without running the matmul.

- [ ] **Step 1: Write the failing test** — `tests/test_compression_toggle.py`

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest loralink_reviewer_response/tests/test_compression_toggle.py -v`
Expected: FAIL — `test_toggle_off_is_lossless` fails (default params still lossy); `test_fake_benchmark` fails (still runs matmul, may pass timing by luck — assert on flops==5.0 fails).

- [ ] **Step 3: Patch `compression_engine.py`**

In `OptimizedCompressionEngine.__init__`, after `self.compression_params = {...}`:

```python
        import os
        if os.environ.get("LORALINK_LOSSY_COMPRESSION", "1") == "0":
            for k in self.compression_params:
                self.compression_params[k] = {"sparsity_ratio": 0.0, "quantize": False}
            logger.info("OptimizedCompressionEngine: lossy compression DISABLED "
                        "(lossless zstd only)")
        else:
            logger.info("OptimizedCompressionEngine: lossy compression ENABLED")
```

(No other change — `sparsify_by_magnitude` already returns the tensor untouched at ratio 0.0, and `quantize=False` skips int8.)

- [ ] **Step 4: Patch `benchmarking.py`**

At the very top of `run_benchmark()`:

```python
    import os
    if os.environ.get("LORALINK_FAKE_BENCHMARK", "0") == "1":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            mem_gb = psutil.virtual_memory().available / (1024 ** 3)
        print("⚠️  FAKE benchmark (LORALINK_FAKE_BENCHMARK=1) — synthetic stats")
        return {"flops": 5.0, "memory_gb": float(mem_gb),
                "device_type": "cuda" if device.type == "cuda" else "cpu"}
```

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest loralink_reviewer_response/tests/test_compression_toggle.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add compression_engine.py benchmarking.py loralink_reviewer_response/tests/test_compression_toggle.py
git commit -m "feat: env toggles for lossless compression + fast benchmark"
```

---

## Task 5: `data_loader.py` — evaluation hold-out split

**Files:**
- Modify: `data_loader.py` (`get_data_loader`)
- Test: `loralink_reviewer_response/tests/test_data_holdout.py`

**Interfaces:**
- Consumes: existing `DATASET_REGISTRY`, `get_data_loader` signature
- Produces: `get_data_loader(tokenizer, num_samples=5, dataset_name="wikitext", *, split="train", eval_holdout=0)`:
  - `split="train"` + `eval_holdout=K` → training loader uses samples `[0 : len-K]` (unchanged behaviour when `eval_holdout=0`).
  - `split="eval"` → loader uses the last `min(num_samples, K_or_validation)` samples: the dataset's `validation`/`test` split if the HF dataset defines one, else the final `num_samples` rows of `train` (disjoint from training when the same `eval_holdout` is passed to both calls).
  - Prints and returns-via-attribute the exact slice bounds: `loader.slice_bounds = (start, end, source_split)`.

- [ ] **Step 1: Write the failing test** — `tests/test_data_holdout.py` **[local]** (uses a tiny fake dataset, no network)

```python
import torch
from transformers import AutoTokenizer
import data_loader

class _FakeDS(list):
    column_names = ["input_ids", "attention_mask"]
    def select(self, rng): return _FakeDS(self[i] for i in rng)
    def set_format(self, *a, **k): pass
    def map(self, *a, **k): return self
    def filter(self, *a, **k): return self
    def __len__(self): return list.__len__(self)

def test_train_and_eval_slices_are_disjoint(monkeypatch):
    rows = [{"input_ids": torch.tensor([i]), "attention_mask": torch.tensor([1])}
            for i in range(20)]
    monkeypatch.setattr(data_loader, "load_dataset", lambda **k: _FakeDS(rows))
    tok = AutoTokenizer.from_pretrained("gpt2")           # cached in CI image
    tr = data_loader.get_data_loader(tok, num_samples=15, dataset_name="wikitext",
                                     split="train", eval_holdout=5)
    ev = data_loader.get_data_loader(tok, num_samples=5, dataset_name="wikitext",
                                     split="eval", eval_holdout=5)
    tr_ids = {int(b["input_ids"][0]) for b in tr}
    ev_ids = {int(b["input_ids"][0]) for b in ev}
    assert tr_ids.isdisjoint(ev_ids)
    assert ev.slice_bounds[0] >= 15
```

(If mocking `datasets` internals proves brittle, downgrade this to a **[colab]** test that calls the real `wikitext` loader and just asserts `tr_ids.isdisjoint(ev_ids)` and `len(ev) == 5`.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest loralink_reviewer_response/tests/test_data_holdout.py -v`
Expected: FAIL — `get_data_loader() got an unexpected keyword argument 'split'`

- [ ] **Step 3: Patch `data_loader.py`**

- Add `*, split: str = "train", eval_holdout: int = 0` to the signature; keep the asserts.
- After `tokenized_dataset` is built and formatted, replace the `subset_dataset = ...` line with:

```python
    total = len(tokenized_dataset)
    if split == "eval":
        val = None
        for cand in ("validation", "test"):
            try:
                v = load_dataset(**{**load_args, "split": cand})
                v = v.map(ds_info["formatter"], batched=True)
                v = v.filter(lambda x: len(x["text"].strip()) > 0)
                v = v.map(tokenize_function, batched=True, remove_columns=remove_cols)
                v.set_format(type="torch", columns=["input_ids", "attention_mask"])
                val = v
                break
            except Exception:
                continue
        if val is not None:
            end = min(num_samples, len(val))
            subset_dataset = val.select(range(end))
            bounds = (0, end, cand)
        else:
            start = max(0, total - max(num_samples, eval_holdout))
            end = min(total, start + num_samples)
            subset_dataset = tokenized_dataset.select(range(start, end))
            bounds = (start, end, "train-tail")
    else:
        end = total - eval_holdout if eval_holdout else min(num_samples, total)
        end = max(1, min(end, total, num_samples if not eval_holdout else end))
        subset_dataset = tokenized_dataset.select(range(0, end))
        bounds = (0, end, "train")
    print(f"📐 slice bounds: {bounds}")
```

- After the `DataLoader` is constructed: `dataloader.slice_bounds = bounds`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest loralink_reviewer_response/tests/test_data_holdout.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data_loader.py loralink_reviewer_response/tests/test_data_holdout.py
git commit -m "feat: disjoint eval hold-out split in data_loader"
```

---

## Task 6: `main.py` — CLI flags, seeding, bind address, epoch loop, metrics

**Files:**
- Modify: `main.py` (argparse block ≈ 860–883; `run_coordinator` ≈ 424–648; `run_worker` ≈ 845–858)
- Test: `loralink_reviewer_response/tests/test_main_cli.py` **[local]** (argparse only)

**Interfaces:**
- Consumes: `metrics_logger.append_rows`, `RUN_COLUMNS`, `SUMMARY_COLUMNS`; `data_loader.get_data_loader(..., split, eval_holdout)`; `DeviceManager.partition_model(master_ip, strategy)`
- Produces: `main.py` CLI accepts and threads:
  - `--seed INT` (default 42) — seeds `random`, `numpy`, `torch`, `torch.cuda` in both roles before any model construction; also stored into `model_config["seed"]` for `random` strategy determinism.
  - `--num-samples INT` (default 60) — passed to `get_data_loader`.
  - `--epochs INT` (default 1) — wraps the batch loop; `global_batch` counter spans epochs.
  - `--eval-holdout INT` (default 0) — passed to `get_data_loader`.
  - `--partition-strategy {smart,round_robin,proportional,random}` (default `smart`) — passed to `partition_model`.
  - `--base-model STR` — alias accepted for `--model-path` (keep `--model-path` working; `--base-model` wins if both given).
  - `--run-tag STR` (default `""`), `--metrics-csv PATH` (default `""` → disabled).
  - `--host-ip` honoured by the `worker` role too.
  - `NetworkManager` binds `args.host_ip or "0.0.0.0"` in both roles (port stays 29500).
  - Per-batch `RUN_COLUMNS` row + one `SUMMARY_COLUMNS` row appended to `--metrics-csv` when set.

- [ ] **Step 1: Write the failing test** — `tests/test_main_cli.py`

```python
import subprocess, sys

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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest loralink_reviewer_response/tests/test_main_cli.py -v`
Expected: FAIL — new flags absent from `--help`

- [ ] **Step 3: Extend the argparse block** (`main.py` ≈ 860)

```python
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-holdout", type=int, default=0)
    parser.add_argument("--partition-strategy",
                        choices=["smart", "round_robin", "proportional", "random"],
                        default="smart")
    parser.add_argument("--base-model", type=str, default=None,
                        help="alias for --model-path; wins if both set")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--metrics-csv", type=str, default="")
```

Immediately after `args = parser.parse_args()`:

```python
    if args.base_model:
        args.model_path = args.base_model

    import random as _random, numpy as _np
    _random.seed(args.seed); _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"🎲 seed={args.seed}")
```

- [ ] **Step 4: Bind address + strategy + seed-in-config** (`run_coordinator`)

- Line ≈ 445: `NETWORK_MANAGER = NetworkManager(args.host_ip or "0.0.0.0", 29500, _coordinator_message_handler)`
- Line ≈ 449: `model_config = {"model_name": args.model_path, "seed": args.seed}`
- Line ≈ 480: `configs = DEVICE_MANAGER.partition_model(args.host_ip, strategy=args.partition_strategy)`

- [ ] **Step 5: `run_worker` — accept host-ip + seed** (`main.py` ≈ 845)

```python
def run_worker(args):
    global NETWORK_MANAGER
    print("✅ Starting worker node...")
    NETWORK_MANAGER = NetworkManager(args.host_ip or "0.0.0.0", 29500, _worker_message_handler)
    NETWORK_MANAGER.start_server()
    ...
```

(Seeding already done in `__main__` before `run_worker` is called.)

- [ ] **Step 6: Epoch loop + `num_samples` + eval-holdout + metrics** (`run_coordinator` training section ≈ 568–588)

Replace the data-loader call and the `for batch_idx, batch in enumerate(train_loader):` block with:

```python
    train_loader = data_loader.get_data_loader(
        tokenizer, num_samples=args.num_samples, dataset_name=args.dataset,
        split="train", eval_holdout=args.eval_holdout)

    from loralink_reviewer_response.metrics_logger import (
        append_rows, RUN_COLUMNS, SUMMARY_COLUMNS)
    import time as _time
    _run_start = _time.perf_counter()
    _losses, _latencies = [], []
    _sim = "loopback" if (args.host_ip or "").startswith("127.") else "real"
    global_batch = 0

    for epoch in range(args.epochs):
        for batch in train_loader:
            t0 = _time.perf_counter()
            PIPELINE_ENGINE.forward_step_local(global_batch, batch)
            try:
                rid, gradient, loss_value = GRADIENT_QUEUE.get(timeout=300)
            except queue.Empty:
                print(f"❌ Timeout waiting for gradient for batch {global_batch}")
                sys.exit(1)
            assert rid == global_batch, "❌ Mismatched batch ID!"
            PIPELINE_ENGINE.backward_step(rid, gradient)
            step_latency = _time.perf_counter() - t0
            if loss_value is not None:
                _losses.append(float(loss_value))
                print(f"✅ loss[{epoch}:{global_batch}] = {loss_value:.4f}")
            _latencies.append(step_latency)

            if args.metrics_csv:
                cs = PIPELINE_ENGINE.compression_engine.get_compression_stats()
                ratio = cs.get("average_compression_ratio", "").rstrip("x") or ""
                append_rows(args.metrics_csv, [{
                    "run_tag": args.run_tag, "seed": args.seed,
                    "strategy": args.partition_strategy,
                    "compression": os.environ.get("LORALINK_LOSSY_COMPRESSION", "1"),
                    "dataset": args.dataset, "model": args.model_path,
                    "n_workers": len(ACTIVE_WORKER_IPS), "epoch": epoch,
                    "global_batch": global_batch,
                    "loss": "" if loss_value is None else float(loss_value),
                    "step_latency_s": step_latency, "comp_ratio": ratio,
                    "bytes_sent": PIPELINE_ENGINE.compression_engine.stats["total_compressed_bytes"],
                    "bytes_saved": (PIPELINE_ENGINE.compression_engine.stats["total_original_bytes"]
                                    - PIPELINE_ENGINE.compression_engine.stats["total_compressed_bytes"]),
                    "sim": _sim, "timestamp": _time.time(),
                }], RUN_COLUMNS)
            global_batch += 1
```

After "Training complete" (before weight collection), append the summary row:

```python
    if args.metrics_csv:
        st = PIPELINE_ENGINE.compression_engine.stats
        counts = [len(c.assigned_layers) for c in configs.values()]
        import statistics as _stx
        append_rows(args.metrics_csv, [{
            "run_tag": args.run_tag, "seed": args.seed,
            "strategy": args.partition_strategy,
            "compression": os.environ.get("LORALINK_LOSSY_COMPRESSION", "1"),
            "dataset": args.dataset, "model": args.model_path,
            "n_workers": len(ACTIVE_WORKER_IPS), "sim": _sim,
            "n_batches": global_batch,
            "mean_loss": (sum(_losses)/len(_losses)) if _losses else "",
            "last_loss": _losses[-1] if _losses else "",
            "mean_step_latency_s": (sum(_latencies)/len(_latencies)) if _latencies else "",
            "total_bytes_sent": st["total_compressed_bytes"],
            "total_bytes_saved": st["total_original_bytes"] - st["total_compressed_bytes"],
            "overall_comp_ratio": (st["total_original_bytes"] / st["total_compressed_bytes"])
                                   if st["total_compressed_bytes"] else "",
            "wall_time_s": _time.perf_counter() - _run_start,
            "partition_map": ";".join(f"{ip}:{len(c.assigned_layers)}"
                                      for ip, c in configs.items()),
            "partition_balance_std": _stx.pstdev(counts) if len(counts) > 1 else 0.0,
        }], SUMMARY_COLUMNS)
```

Add `import os` at module top if not already effectively imported (it is, line 22).

- [ ] **Step 7: Make `loralink_reviewer_response` importable from `main.py`**

`cluster_launch.py` sets `PYTHONPATH` to the repo root when spawning `main.py`, so `from loralink_reviewer_response.metrics_logger import ...` resolves. Add a `loralink_reviewer_response/__init__.py` (empty). Note this in `README.md`.

- [ ] **Step 8: Run the CLI test**

Run: `python -m pytest loralink_reviewer_response/tests/test_main_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 9: Commit**

```bash
git add main.py loralink_reviewer_response/__init__.py loralink_reviewer_response/tests/test_main_cli.py
git commit -m "feat: main.py experiment flags, seeding, bind-ip, epochs, metrics"
```

---

## Task 7: `patch/` — diffs, applier, checksums

**Files:**
- Create: `loralink_reviewer_response/patch/apply_patch.py`, `loralink_reviewer_response/patch/checksums.py`, `loralink_reviewer_response/patch/*.patch` (5 files), `loralink_reviewer_response/patch/README.md`
- Test: `loralink_reviewer_response/tests/test_patch_roundtrip.py` **[local]**

**Interfaces:**
- Consumes: the five modified source files from Tasks 3–6
- Produces:
  - `patch/apply_patch.py --repo <dir>` — copies pristine sources into `<dir>`, applies the 5 diffs (via `patch` or a pure-Python fallback), runs `checksums.py --verify`.
  - `patch/checksums.py --update` writes `patch/SHA256SUMS`; `--verify` exits non-zero on mismatch.
  - `patch/MANIFEST.json` — `{file: {sha256_before, sha256_after}}` for the five files.

- [ ] **Step 1: Generate the diffs**

If git accepted:
```bash
git diff HEAD~5 -- main.py > loralink_reviewer_response/patch/main.py.patch
git diff HEAD~5 -- device_manager.py > loralink_reviewer_response/patch/device_manager.py.patch
git diff HEAD~5 -- compression_engine.py > loralink_reviewer_response/patch/compression_engine.py.patch
git diff HEAD~5 -- benchmarking.py > loralink_reviewer_response/patch/benchmarking.py.patch
git diff HEAD~5 -- data_loader.py > loralink_reviewer_response/patch/data_loader.py.patch
```
If no git: keep a pristine copy of the repo made in Task 0 Step 1 (`cp -r` to `loralink_reviewer_response/patch/_pristine/`) and `diff -u _pristine/<f> <f>` for each.

- [ ] **Step 2: Write `checksums.py`**

```python
"""Record / verify SHA-256 of the five patched LoraLink source files."""
import hashlib, json, sys, pathlib
FILES = ["main.py", "device_manager.py", "compression_engine.py",
         "benchmarking.py", "data_loader.py"]

def _sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()

def main():
    root = pathlib.Path(__file__).resolve().parents[2]
    sums = {f: _sha(root / f) for f in FILES}
    out = pathlib.Path(__file__).with_name("SHA256SUMS")
    if "--update" in sys.argv:
        out.write_text(json.dumps(sums, indent=2)); print("wrote", out); return
    if "--verify" in sys.argv:
        want = json.loads(out.read_text())
        bad = [f for f in FILES if want.get(f) != sums[f]]
        if bad:
            print("CHECKSUM MISMATCH:", bad); sys.exit(1)
        print("checksums OK")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write `apply_patch.py`** (pure-Python unified-diff applier fallback so it works on Colab without `patch(1)` — use `patch-ng` if `requirements-colab.txt` includes it, else shell `patch`)

```python
"""Apply the reviewer-response source patches to a fresh repo copy."""
import argparse, shutil, subprocess, sys, pathlib
PATCH_DIR = pathlib.Path(__file__).resolve().parent
SRC_ROOT = PATCH_DIR.parents[1]
FILES = ["main.py", "device_manager.py", "compression_engine.py",
         "benchmarking.py", "data_loader.py"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    a = ap.parse_args()
    dst = pathlib.Path(a.repo)
    # In Colab the repo IS already the patched checkout (we ship patched
    # sources in the package). apply_patch just verifies integrity.
    r = subprocess.run([sys.executable, str(PATCH_DIR / "checksums.py"), "--verify"])
    sys.exit(r.returncode)

if __name__ == "__main__":
    main()
```

Decision: **ship the patched sources directly** in the Colab flow (the notebook `git clone`s the repo *with the patch already merged in* — i.e. this whole working tree is the deliverable). `apply_patch.py` degrades to a checksum verifier. The `.patch` files are kept for reviewers who want to see exactly what changed vs the paper's code. Document this in `patch/README.md`.

- [ ] **Step 4: Write the failing test** — `tests/test_patch_roundtrip.py`

```python
import subprocess, sys

def test_checksums_verify_clean(pkg_dir):
    subprocess.run([sys.executable, str(pkg_dir/"patch"/"checksums.py"), "--update"], check=True)
    r = subprocess.run([sys.executable, str(pkg_dir/"patch"/"checksums.py"), "--verify"])
    assert r.returncode == 0

def test_patch_files_exist_and_nonempty(pkg_dir):
    for f in ["main.py", "device_manager.py", "compression_engine.py",
              "benchmarking.py", "data_loader.py"]:
        p = pkg_dir / "patch" / f"{f}.patch"
        assert p.exists() and p.stat().st_size > 0

def test_patches_mention_only_allowed_changes(pkg_dir):
    banned = ["optimizer", "AdamW", "lr=", "learning_rate", "loss =",
              "backward()", "sparsify_by_magnitude(", "quantize_to_int8("]
    for f in ["compression_engine.py", "device_manager.py"]:
        txt = (pkg_dir/"patch"/f"{f}.patch").read_text().lower()
        added = "\n".join(l for l in txt.splitlines() if l.startswith("+"))
        for b in banned:
            assert b.lower() not in added, (f, b)
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest loralink_reviewer_response/tests/test_patch_roundtrip.py -v`
Expected: PASS (3 passed). If `test_patches_mention_only_allowed_changes` trips on a legitimate context line, tighten the added-line filter to ignore `+++ ` headers.

- [ ] **Step 6: Commit**

```bash
git add loralink_reviewer_response/patch loralink_reviewer_response/tests/test_patch_roundtrip.py
git commit -m "feat: patch bundle + checksum provenance"
```

---

## Task 8: `cluster_launch.py`

**Files:**
- Create: `loralink_reviewer_response/cluster_launch.py`
- Test: `loralink_reviewer_response/tests/test_cluster_smoke.py` **[colab]** (needs transformers + a model; also runnable locally CPU with `gpt-neo-125M` cached)

**Interfaces:**
- Consumes: patched `main.py` CLI; `metrics_logger` (indirectly, via `main.py`)
- Produces:
  - `run_cluster(n_workers, dataset, seed, *, model="EleutherAI/gpt-neo-125M", strategy="smart", compression=True, num_samples=60, epochs=1, eval_holdout=0, netem=None, tag="", run_timeout_s=900, results_csv="results.csv", save_adapters_to=None, workdir=".") -> str` — returns `results_csv` path. Raises `TimeoutError` (killing children) past `run_timeout_s`.
  - `netem` dict: `{"delay_ms": int, "loss_pct": float, "rate": str|None}` → applied to `lo` via `tc` if root+`tc` available, else an in-process monkeypatch shim on `network_protocol.NetworkManager.send_message` that `time.sleep`s `delay_ms` and drops `loss_pct`% of sends (raising to trigger the retry path). The chosen mechanism is recorded in the returned CSV's `sim` interplay + a `netem_mode` note file next to the CSV.
  - IP plan: coordinator `127.0.0.1`, workers `127.0.0.2 .. 127.0.0.(n+1)`.
  - Process model: `subprocess.Popen([sys.executable, "main.py", "--role", "worker", "--host-ip", ip, "--base-model", model])` per worker (env: `LORALINK_LOSSY_COMPRESSION`, `LORALINK_FAKE_BENCHMARK`, `PYTHONPATH=<repo root>`); wait for each worker's port to accept a TCP connect; then run the coordinator **as a subprocess too** (cleaner signal handling) with `--workers 127.0.0.2,...,--host-ip 127.0.0.1 --metrics-csv <path> --run-tag <tag> --partition-strategy <s> --seed <seed> --num-samples <n> --epochs <e> --dataset <d> --eval-holdout <k>`.
  - Teardown: SIGTERM then SIGKILL every child in a `finally`; `tc qdisc del dev lo root` if it was added.

- [ ] **Step 1: Write the failing smoke test** — `tests/test_cluster_smoke.py`

```python
import pytest, pandas as pd
from loralink_reviewer_response.cluster_launch import run_cluster

@pytest.mark.colab
def test_two_worker_run_writes_rows(tmp_path):
    csv = run_cluster(
        n_workers=2, dataset="wikitext", seed=0,
        model="EleutherAI/gpt-neo-125M", num_samples=6, epochs=1,
        tag="smoke", results_csv=str(tmp_path / "r.csv"),
        run_timeout_s=600, workdir=".")
    df = pd.read_csv(csv)
    assert (df["run_tag"] == "smoke").any()
    assert df["loss"].dropna().shape[0] >= 3          # got some losses
    assert df["step_latency_s"].dropna().gt(0).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest loralink_reviewer_response/tests/test_cluster_smoke.py -v -m colab`
Expected: FAIL — `ModuleNotFoundError: cluster_launch`

- [ ] **Step 3: Implement `cluster_launch.py`**

Full implementation (no placeholders — write it out):

```python
"""Spawn a LoraLink coordinator + N workers on loopback for one experiment run."""
from __future__ import annotations
import os, sys, socket, time, signal, shutil, subprocess, contextlib, pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

def _wait_port(ip, port, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        with contextlib.suppress(OSError):
            with socket.create_connection((ip, port), timeout=1):
                return True
        time.sleep(0.5)
    raise TimeoutError(f"{ip}:{port} never came up")

def _tc_available():
    return shutil.which("tc") is not None and os.geteuid() == 0

def _apply_netem(netem):
    parts = ["tc", "qdisc", "add", "dev", "lo", "root", "netem"]
    if netem.get("delay_ms"): parts += ["delay", f'{netem["delay_ms"]}ms']
    if netem.get("loss_pct"): parts += ["loss", f'{netem["loss_pct"]}%']
    if netem.get("rate"):     parts += ["rate", netem["rate"]]
    subprocess.run(parts, check=True)

def _clear_netem():
    subprocess.run(["tc", "qdisc", "del", "dev", "lo", "root"],
                   check=False, capture_output=True)

def run_cluster(n_workers, dataset, seed, *, model="EleutherAI/gpt-neo-125M",
                strategy="smart", compression=True, num_samples=60, epochs=1,
                eval_holdout=0, netem=None, tag="", run_timeout_s=900,
                results_csv="results.csv", save_adapters_to=None, workdir="."):
    workdir = pathlib.Path(workdir).resolve()
    coord_ip = "127.0.0.1"
    worker_ips = [f"127.0.0.{i}" for i in range(2, 2 + n_workers)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["LORALINK_LOSSY_COMPRESSION"] = "1" if compression else "0"
    env.setdefault("LORALINK_FAKE_BENCHMARK", env.get("LORALINK_FAKE_BENCHMARK", "0"))

    netem_mode = "none"
    procs = []
    try:
        if netem:
            if _tc_available():
                _apply_netem(netem); netem_mode = "tc-netem"
            else:
                netem_mode = "in-process-shim"
                env["LORALINK_NET_SHIM"] = f'{netem.get("delay_ms",0)},{netem.get("loss_pct",0)}'
        (workdir / (results_csv + ".netem")).write_text(netem_mode)

        for ip in worker_ips:
            procs.append(subprocess.Popen(
                [sys.executable, "main.py", "--role", "worker",
                 "--host-ip", ip, "--base-model", model, "--seed", str(seed)],
                cwd=workdir, env=env))
        for ip in worker_ips:
            _wait_port(ip, 29500)

        coord = subprocess.Popen(
            [sys.executable, "main.py", "--role", "coordinator",
             "--host-ip", coord_ip, "--workers", ",".join(worker_ips),
             "--base-model", model, "--dataset", dataset, "--seed", str(seed),
             "--num-samples", str(num_samples), "--epochs", str(epochs),
             "--eval-holdout", str(eval_holdout),
             "--partition-strategy", strategy, "--run-tag", tag,
             "--metrics-csv", str(workdir / results_csv)],
            cwd=workdir, env=env)
        procs.append(coord)
        try:
            coord.wait(timeout=run_timeout_s)
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"run '{tag}' exceeded {run_timeout_s}s")
        if coord.returncode != 0:
            raise RuntimeError(f"coordinator exited {coord.returncode} for run '{tag}'")

        if save_adapters_to:
            src = workdir / "lora_adapters"
            if src.exists():
                shutil.copytree(src, save_adapters_to, dirs_exist_ok=True)
        return str(workdir / results_csv)
    finally:
        for p in procs:
            with contextlib.suppress(Exception):
                p.send_signal(signal.SIGTERM)
        time.sleep(2)
        for p in procs:
            with contextlib.suppress(Exception):
                p.kill()
        if netem_mode == "tc-netem":
            _clear_netem()
```

- [ ] **Step 4: Add the in-process net shim to the patch** (only if `tc` unavailable)

In `network_protocol.py` `send_message`, at the very top of the method body, add (this is a 6th patched file — update `checksums.py` `FILES` and Task 7 accordingly, or fold into `main.py` import time monkeypatch to avoid touching `network_protocol.py`). **Preferred:** monkeypatch from `main.py.__main__` when `LORALINK_NET_SHIM` is set, so `network_protocol.py` stays untouched:

```python
    shim = os.environ.get("LORALINK_NET_SHIM")
    if shim:
        import network_protocol, random as _r, time as _t
        d_ms, loss = shim.split(","); d = float(d_ms)/1000.0; loss = float(loss)/100.0
        _orig = network_protocol.NetworkManager.send_message.__func__
        def _shim(peer_ip, peer_port, message):
            if d: _t.sleep(d)
            if loss and _r.random() < loss:
                raise ConnectionError("net-shim simulated drop")
            return _orig(peer_ip, peer_port, message)
        network_protocol.NetworkManager.send_message = staticmethod(_shim)
```

Add this block to Task 6 Step 3 (after seeding) and note it there. `network_protocol.py` is **not** patched.

- [ ] **Step 5: Run the smoke test**

Run: `python -m pytest loralink_reviewer_response/tests/test_cluster_smoke.py -v -m colab` (in Colab or locally with `gpt-neo-125M` downloaded)
Expected: PASS — CSV has ≥ 3 loss rows, all `step_latency_s > 0`.

- [ ] **Step 6: Commit**

```bash
git add loralink_reviewer_response/cluster_launch.py main.py loralink_reviewer_response/tests/test_cluster_smoke.py
git commit -m "feat: localhost cluster launcher with netem + timeout"
```

---

## Task 9: `eval_quality.py`

**Files:**
- Create: `loralink_reviewer_response/eval_quality.py`
- Test: `loralink_reviewer_response/tests/test_eval_quality.py` **[colab]**

**Interfaces:**
- Consumes: a saved adapter dir (`./lora_adapters` from a run), `peft`, `evaluate`, patched `data_loader.get_data_loader(..., split="eval", eval_holdout=K)`
- Produces:
  - `evaluate_adapter(base_model: str, adapter_dir: str, dataset: str, *, eval_holdout=200, max_new_tokens=48, limit=100, arm="", seed=0, out_csv="results_quality.csv") -> dict` with keys `perplexity`, `bleu`, `rougeL`, `n_eval`, plus the passthrough tags. Appends one row (columns: `arm, seed, dataset, base_model, perplexity, bleu, rougeL, n_eval, adapter_dir, slice_bounds`).
  - Perplexity: `exp(mean token NLL)` over the eval slice (teacher-forced, `labels=input_ids`, ignore pad).
  - BLEU/ROUGE (E2E only): greedy-decode from the `meaning_representation` prompt prefix, compare to `target` references via `evaluate.load("sacrebleu")` and `evaluate.load("rouge")`. For WikiText, `bleu = rougeL = ""` (PPL only).

- [ ] **Step 1: Write the failing test** — `tests/test_eval_quality.py`

```python
import pytest
from loralink_reviewer_response.eval_quality import evaluate_adapter

@pytest.mark.colab
def test_eval_runs_on_base_only(tmp_path):
    # adapter_dir="" -> evaluate the base model with no adapter (sanity path)
    r = evaluate_adapter("EleutherAI/gpt-neo-125M", "", "wikitext",
                         eval_holdout=20, limit=10,
                         out_csv=str(tmp_path/"q.csv"))
    assert r["perplexity"] > 1.0 and r["n_eval"] == 10
    assert r["bleu"] == "" and r["rougeL"] == ""
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest loralink_reviewer_response/tests/test_eval_quality.py -v -m colab`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `eval_quality.py`** (write in full — no placeholders)

```python
"""Held-out perplexity + BLEU/ROUGE for a LoRA adapter (or bare base model)."""
from __future__ import annotations
import math, csv, os, pathlib, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

_COLS = ["arm", "seed", "dataset", "base_model", "perplexity", "bleu",
         "rougeL", "n_eval", "adapter_dir", "slice_bounds"]

def _load(base_model, adapter_dir):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float32)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tok

@torch.no_grad()
def _perplexity(model, tok, texts, max_length=256):
    dev = next(model.parameters()).device
    nll, ntok = 0.0, 0
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc.input_ids.to(dev)
        if ids.numel() < 2:
            continue
        out = model(ids, labels=ids)
        n = ids.numel() - 1
        nll += out.loss.item() * n
        ntok += n
    return math.exp(nll / max(ntok, 1))

@torch.no_grad()
def _gen_bleu_rouge(model, tok, mrs, refs, max_new_tokens):
    import evaluate as _ev
    dev = next(model.parameters()).device
    preds = []
    for mr in mrs:
        prompt = f"Data: {mr}\nText:"
        enc = tok(prompt, return_tensors="pt").to(dev)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens,
                             do_sample=False, pad_token_id=tok.pad_token_id)
        preds.append(tok.decode(gen[0][enc.input_ids.shape[1]:],
                                skip_special_tokens=True).strip())
    bleu = _ev.load("sacrebleu").compute(
        predictions=preds, references=[[r] for r in refs])["score"]
    rouge = _ev.load("rouge").compute(
        predictions=preds, references=refs)["rougeL"]
    return bleu, rouge * 100.0

def evaluate_adapter(base_model, adapter_dir, dataset, *, eval_holdout=200,
                     max_new_tokens=48, limit=100, arm="", seed=0,
                     out_csv="results_quality.csv"):
    model, tok = _load(base_model, adapter_dir)
    if dataset == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test",
                          cache_dir="./dataset")
        texts = [x for x in ds["text"] if x.strip()][:limit]
        ppl = _perplexity(model, tok, texts)
        bleu = rougeL = ""
        n_eval = len(texts); bounds = f"test[0:{len(texts)}]"
    else:  # e2e
        ds = load_dataset("GEM/e2e_nlg", split="validation",
                          cache_dir="./dataset", trust_remote_code=True)
        ds = ds.select(range(min(limit, len(ds))))
        mrs = ds["meaning_representation"]; refs = ds["target"]
        texts = [f"Data: {m}\nText: {t}" for m, t in zip(mrs, refs)]
        ppl = _perplexity(model, tok, texts)
        bleu, rougeL = _gen_bleu_rouge(model, tok, mrs, refs, max_new_tokens)
        n_eval = len(ds); bounds = f"validation[0:{len(ds)}]"

    row = {"arm": arm, "seed": seed, "dataset": dataset,
           "base_model": base_model, "perplexity": round(ppl, 4),
           "bleu": bleu if bleu == "" else round(bleu, 3),
           "rougeL": rougeL if rougeL == "" else round(rougeL, 3),
           "n_eval": n_eval, "adapter_dir": adapter_dir or "(base)",
           "slice_bounds": bounds}
    p = pathlib.Path(out_csv)
    need = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        if need: w.writeheader()
        w.writerow(row)
    return row
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest loralink_reviewer_response/tests/test_eval_quality.py -v -m colab`
Expected: PASS — base-model PPL is a finite number > 1, CSV row written.

- [ ] **Step 5: Commit**

```bash
git add loralink_reviewer_response/eval_quality.py loralink_reviewer_response/tests/test_eval_quality.py
git commit -m "feat: held-out PPL + BLEU/ROUGE adapter eval"
```

---

## Task 10: `baselines/` — published numbers

**Files:**
- Create: `loralink_reviewer_response/baselines/published_baselines.csv`
- Create: `loralink_reviewer_response/baselines/SOURCES.md`
- Test: `loralink_reviewer_response/tests/test_baselines.py` **[local]**

**Interfaces:**
- Consumes: nothing (literature research)
- Produces: `published_baselines.csv` columns `method,source_ref,model,params,dataset,metric,value,unit,hardware,comparable,notes` where `comparable ∈ {direct, trend, context}`; `SOURCES.md` has one `## [ref]` block per distinct `source_ref` with full citation, URL/DOI, exact table/section, and a `> verbatim` quote of the number.

- [ ] **Step 1: Research and record the numbers** (use WebSearch / WebFetch on arXiv)

Gather, at minimum, these rows (add more if a clean number exists). For each, copy the *exact* figure and its location:

| method | typical numbers to capture | source |
|---|---|---|
| SplitLoRA | communication volume per step / time-to-target-PPL, GPT-2 or LLaMA-7B, E2E or WikiText | Lin et al., "SplitLoRA", arXiv 2407.00952 |
| HSplitLoRA | perplexity / throughput vs SplitLoRA & centralized, LLaMA | "HSplitLoRA" arXiv 2505.02795 (verify id) |
| QLoRA | memory (GB) to finetune 7B/13B, MMLU or Vicuna score vs 16-bit LoRA | Dettmers et al., arXiv 2305.14314, Tables 3–4 |
| Petals | fine-tune / inference throughput (tokens/s) over the internet, BLOOM-176B / LLaMA-65B | Borzunov et al., arXiv 2209.01188 |
| DeepSpeed ZeRO | throughput (TFLOPs) / max model size vs data-parallel, scaling curve | Rajbhandari et al., arXiv 1910.02054 |
| PyTorch FSDP | TFLOPs/GPU, scaling efficiency, 175B/1T | Zhao et al., arXiv 2304.11277 |
| Megatron-LM | scaling efficiency % at N GPUs (pipeline+tensor parallel) | Narayanan et al., arXiv 2104.04473, Fig/Table on efficiency |

Mark `comparable`:
- `direct` — SplitLoRA, HSplitLoRA, Petals (commodity / split-learning regime, closest to LoraLink).
- `context` — QLoRA (memory-efficiency reference point, single-GPU).
- `trend` — DeepSpeed, FSDP, Megatron (data-centre scale; cited for the scaling-efficiency trend only, explicitly not head-to-head).

- [ ] **Step 2: Write `published_baselines.csv`** — real rows, e.g.:

```csv
method,source_ref,model,params,dataset,metric,value,unit,hardware,comparable,notes
SplitLoRA,lin2024splitlora,LLaMA,7B,E2E NLG,communication_per_round,<FILL>,MB,"2x RTX 3090",direct,"server-client split; number from Table <FILL>"
QLoRA,dettmers2023qlora,LLaMA,7B,finetune,peak_memory,<FILL>,GB,"1x A100 48GB",context,"4-bit NF4; Sec 4 / Table <FILL>"
...
```

Every `<FILL>` must be replaced with a transcribed value in Step 1; the test below fails while any `<FILL>` remains.

- [ ] **Step 3: Write the failing test** — `tests/test_baselines.py`

```python
import csv, re, pathlib

def test_no_unfilled_placeholders(pkg_dir):
    txt = (pkg_dir/"baselines"/"published_baselines.csv").read_text()
    assert "<FILL>" not in txt and "TODO" not in txt

def test_every_row_has_a_source_block(pkg_dir):
    rows = list(csv.DictReader(open(pkg_dir/"baselines"/"published_baselines.csv")))
    src = (pkg_dir/"baselines"/"SOURCES.md").read_text()
    assert len(rows) >= 6
    for r in rows:
        assert r["comparable"] in {"direct", "trend", "context"}
        assert f'[{r["source_ref"]}]' in src or f'## {r["source_ref"]}' in src
        assert float(re.sub(r"[^0-9.\-]", "", r["value"]))  # value is numeric-ish

def test_sources_have_urls_and_quotes(pkg_dir):
    src = (pkg_dir/"baselines"/"SOURCES.md").read_text()
    assert src.count("arxiv.org") + src.count("doi.org") >= 5
    assert src.count(">") >= 6         # at least one verbatim quote per source
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest loralink_reviewer_response/tests/test_baselines.py -v`
Expected: PASS once all numbers are transcribed.

- [ ] **Step 5: Commit**

```bash
git add loralink_reviewer_response/baselines loralink_reviewer_response/tests/test_baselines.py
git commit -m "docs: curated published-baseline table with verbatim sources"
```

---

## Task 11: `aggregate.py` + `statlib` wiring

**Files:**
- Create: `loralink_reviewer_response/aggregate.py`
- Test: `loralink_reviewer_response/tests/test_aggregate.py` **[local]** (synthetic CSVs)

> **AMENDMENT (post-Task-6 review):** `main.py` writes two files per run — per-batch `RUN_COLUMNS` rows to `<metrics_csv>` (e.g. `results_stat_<tag>.csv`) and the single `SUMMARY_COLUMNS` row to a sibling `<metrics_csv stem>.summary.csv` (e.g. `results_stat_<tag>.summary.csv`). The two schemas differ, so they must not share a file. In this task:
> - Summary-level tables (T1 stat-validation means, T4 scheduling throughput/balance, T5 scalability, T6 network) read `_read_many(results_dir, "<prefix>", suffix=".summary.csv")`.
> - Per-batch / loss-curve data (T1 convergence curve) reads the plain `<prefix>*.csv` (excluding `*.summary.csv` — filter it out in the glob).
> - `_read_many` gains a `suffix=".csv"` param; when `".summary.csv"` is requested, match `*{prefix}*.summary.csv`; when `".csv"` is requested, match `*{prefix}*.csv` and drop names ending `.summary.csv`.
> - Test fixtures write `results_*_<tag>.summary.csv` for summary rows and `results_*_<tag>.csv` for per-batch rows.
> - `results_quality_*` (from `eval_quality.py`, Task 9) is a single self-contained schema — unaffected, stays `results_quality_*.csv`.

**Interfaces:**
- Consumes: `results/*.csv` per-batch RUN rows + `results/*.summary.csv` summary rows (`results_stat_*`, `results_sched_*`, `results_scale_*`, `results_net_*`, `results_converge_*`), plus single-schema `results_quality_*.csv` (Task 9), `baselines/published_baselines.csv`, `statlib.mean_std_ci`
- Produces:
  - `build_all(results_dir, baselines_csv, out_dir) -> dict[str, pandas.DataFrame]` writing:
    - `figures/T1_stat_validation.csv` + `figures/T1_loss_curve.png/.pdf`
    - `figures/T2_quality_vs_compression.csv` + `figures/T2_quality_bars.png/.pdf`
    - `figures/T3_ours_vs_published.csv` + `figures/T3_scatter.png/.pdf`
    - `figures/T4_scheduling.csv` + `figures/T4_bars.png/.pdf`
    - `figures/T5_scalability_sim.csv` + `figures/T5_lines.png/.pdf`
    - `figures/T6_network.csv` + `figures/T6_heatmap.png/.pdf`
    - `figures/summary.json` — every headline number with its `[ours]`/`[published]` tag and `n`
  - Missing input files → that table is skipped with a printed `WARN: no data for T#`; `build_all` still returns.
  - `render_response(summary_json, template_md, out_md)` — string-substitutes `{{key}}` placeholders in `RESPONSE_ABHAY_NIKHIL.md`.
- Each figure caption embeds the provenance string and, for T5/T6, "single-box loopback simulation — not WAN".

- [ ] **Step 1: Write the failing test** — `tests/test_aggregate.py`

```python
import pandas as pd, json
from loralink_reviewer_response.aggregate import build_all

def _write(p, rows): pd.DataFrame(rows).to_csv(p, index=False)

def test_stat_and_quality_tables(tmp_path):
    rdir = tmp_path/"results"; rdir.mkdir()
    fig = tmp_path/"figures"
    _write(rdir/"results_stat_a.csv", [
        {"run_tag":"s","seed":s,"dataset":"wikitext","model":"gpt-neo-125M",
         "mean_loss":3.0+0.1*s,"mean_step_latency_s":0.5,"overall_comp_ratio":10.0,
         "n_workers":2,"sim":"loopback"} for s in range(5)])
    _write(rdir/"results_quality_b.csv", [
        {"arm":a,"seed":0,"dataset":"e2e","base_model":"phi-1_5",
         "perplexity":p,"bleu":bl,"rougeL":rl,"n_eval":100}
        for a,p,bl,rl in [("ON",22.0,4.1,18.0),("OFF",21.6,4.3,18.4),
                          ("reference",21.4,4.5,18.7)]])
    bl = tmp_path/"baselines.csv"
    _write(bl, [{"method":"SplitLoRA","source_ref":"x","model":"LLaMA","params":"7B",
                 "dataset":"E2E NLG","metric":"communication_per_round","value":"120",
                 "unit":"MB","hardware":"2x3090","comparable":"direct","notes":""}])
    out = build_all(str(rdir), str(bl), str(fig))
    assert (fig/"T1_stat_validation.csv").exists()
    assert (fig/"T2_quality_vs_compression.csv").exists()
    s = json.loads((fig/"summary.json").read_text())
    assert s["stat_validation"]["wikitext"]["mean_loss"]["n"] == 5
    # ON vs OFF delta present and tagged
    assert "delta_ppl_on_minus_off" in s["quality"]["e2e"]

def test_missing_inputs_are_skipped(tmp_path):
    (tmp_path/"results").mkdir(); (tmp_path/"figures")
    bl = tmp_path/"b.csv"; pd.DataFrame([{"method":"m","source_ref":"r","model":"x",
        "params":"1B","dataset":"d","metric":"z","value":"1","unit":"u",
        "hardware":"h","comparable":"trend","notes":""}]).to_csv(bl, index=False)
    out = build_all(str(tmp_path/"results"), str(bl), str(tmp_path/"figures"))
    assert out is not None    # no crash on empty results
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest loralink_reviewer_response/tests/test_aggregate.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement `aggregate.py`**

Write it out in full. Key structure (the executor fills the matplotlib calls — each is a standard `fig, ax = plt.subplots()`, plot, `savefig(png)`, `savefig(pdf)`; no placeholders in logic):

```python
"""Merge experiment CSVs + published baselines into reviewer-response tables/plots."""
from __future__ import annotations
import glob, json, os, pathlib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loralink_reviewer_response.statlib import mean_std_ci

PROV = "[ours] = we ran it on Colab Free T4 · [published] = transcribed, see baselines/SOURCES.md"

def _read_many(results_dir, prefix):
    files = sorted(glob.glob(os.path.join(results_dir, f"{prefix}*.csv")))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def _t1(results_dir, fig, summary):
    df = _read_many(results_dir, "results_stat_")
    if df is None:
        print("WARN: no data for T1"); return
    df = df[df["mean_loss"].notna() & (df["mean_loss"] != "")]
    rows, summ = [], {}
    for ds, g in df.groupby("dataset"):
        for col in ["mean_loss", "mean_step_latency_s", "overall_comp_ratio"]:
            vals = pd.to_numeric(g[col], errors="coerce").dropna().tolist()
            st = mean_std_ci(vals)
            rows.append({"dataset": ds, "metric": col, **st})
            summ.setdefault(ds, {})[col] = st
    pd.DataFrame(rows).to_csv(fig / "T1_stat_validation.csv", index=False)
    summary["stat_validation"] = summ
    # loss-curve plot from results_converge_* if present, else per-seed mean_loss bar
    conv = _read_many(results_dir, "results_converge_")
    f, ax = plt.subplots(figsize=(6, 4))
    if conv is not None and "global_batch" in conv:
        for (ds,), g in conv.groupby(["dataset"]):
            g = g.sort_values("global_batch")
            ax.plot(g["global_batch"], pd.to_numeric(g["loss"], errors="coerce"),
                    label=f"{ds} [ours]")
        ax.set_xlabel("mini-batch"); ax.set_ylabel("cross-entropy loss")
    else:
        for ds, g in df.groupby("dataset"):
            ax.scatter(g["seed"], pd.to_numeric(g["mean_loss"], errors="coerce"),
                       label=f"{ds} [ours]")
        ax.set_xlabel("seed"); ax.set_ylabel("mean loss")
    ax.set_title("Convergence / loss stability (gpt-neo-125M) [ours]")
    ax.legend(); f.tight_layout()
    f.savefig(fig / "T1_loss_curve.png", dpi=150); f.savefig(fig / "T1_loss_curve.pdf")
    plt.close(f)

def _t2(results_dir, fig, summary):
    df = _read_many(results_dir, "results_quality_")
    if df is None:
        print("WARN: no data for T2"); return
    out, summ = [], {}
    for ds, g in df.groupby("dataset"):
        agg = {}
        for arm, ga in g.groupby("arm"):
            for m in ["perplexity", "bleu", "rougeL"]:
                vals = pd.to_numeric(ga[m], errors="coerce").dropna().tolist()
                if vals:
                    agg[(arm, m)] = mean_std_ci(vals)
                    out.append({"dataset": ds, "arm": arm, "metric": m,
                                **mean_std_ci(vals)})
        if ("ON", "perplexity") in agg and ("OFF", "perplexity") in agg:
            summ.setdefault(ds, {})["delta_ppl_on_minus_off"] = (
                agg[("ON", "perplexity")]["mean"] - agg[("OFF", "perplexity")]["mean"])
        if ("ON", "bleu") in agg and ("OFF", "bleu") in agg:
            summ.setdefault(ds, {})["delta_bleu_on_minus_off"] = (
                agg[("ON", "bleu")]["mean"] - agg[("OFF", "bleu")]["mean"])
    pd.DataFrame(out).to_csv(fig / "T2_quality_vs_compression.csv", index=False)
    summary["quality"] = summ
    # grouped bar with error bars
    piv = (pd.DataFrame(out)
           .pivot_table(index=["dataset", "metric"], columns="arm", values="mean"))
    f, ax = plt.subplots(figsize=(7, 4))
    piv.plot.bar(ax=ax); ax.set_title("Task quality: compression ON vs OFF vs reference [ours]")
    ax.set_ylabel("value (PPL lower better; BLEU/ROUGE higher better)")
    f.tight_layout(); f.savefig(fig / "T2_quality_bars.png", dpi=150)
    f.savefig(fig / "T2_quality_bars.pdf"); plt.close(f)

# _t3 (ours vs published scatter), _t4 (scheduling bars), _t5 (scalability lines),
# _t6 (network heatmap) follow the same read -> aggregate -> csv -> plot pattern.
# Each writes figures/T#_*.csv and figures/T#_*.png/.pdf and updates summary[...].

def build_all(results_dir, baselines_csv, out_dir):
    fig = pathlib.Path(out_dir); fig.mkdir(parents=True, exist_ok=True)
    summary = {"provenance": PROV}
    _t1(results_dir, fig, summary)
    _t2(results_dir, fig, summary)
    _t3(results_dir, baselines_csv, fig, summary)
    _t4(results_dir, fig, summary)
    _t5(results_dir, fig, summary)
    _t6(results_dir, fig, summary)
    (fig / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary

def render_response(summary_json, template_md, out_md):
    data = json.loads(pathlib.Path(summary_json).read_text())
    flat = {}
    def _walk(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items(): _walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            flat[prefix] = obj
    _walk("", data)
    txt = pathlib.Path(template_md).read_text()
    for k, v in flat.items():
        txt = txt.replace("{{" + k + "}}", str(v))
    pathlib.Path(out_md).write_text(txt)
```

The executor must fully implement `_t3`–`_t6` (no stubs left) following the documented pattern, and the tests in Step 1 plus two more (`test_t4_scheduling`, `test_t5_scalability`, `test_t6_network`) must be added and pass.

- [ ] **Step 4: Extend the test file** with `test_t4_scheduling`, `test_t5_scalability`, `test_t6_network` (synthetic CSVs mirroring the real column sets from Task 6's `SUMMARY_COLUMNS` and Task 12's notebook outputs), then run:

Run: `python -m pytest loralink_reviewer_response/tests/test_aggregate.py -v`
Expected: PASS (all cases)

- [ ] **Step 5: Commit**

```bash
git add loralink_reviewer_response/aggregate.py loralink_reviewer_response/tests/test_aggregate.py
git commit -m "feat: aggregation to comparison tables + plots"
```

---

## Task 12: Notebook template + experiment notebooks

**Files:**
- Create: `loralink_reviewer_response/notebooks/_template.ipynb` and `00`, `01`, `02`, `02b`, `03`, `04`, `05`
- Create: `loralink_reviewer_response/notebooks/build_notebooks.py` (generates the 7 from `_template.ipynb` + per-notebook body snippets — keeps them DRY)
- Test: `loralink_reviewer_response/tests/test_notebooks.py` **[local]** (static checks with `nbformat`)

**Interfaces:**
- Consumes: `cluster_launch.run_cluster`, `eval_quality.evaluate_adapter`
- Produces: 7 notebooks, each with exactly the 5-cell contract from spec §5.2 and a walltime guard. Every notebook: `ACCOUNT_TAG`, `SHARD`, `WALL_BUDGET_MIN=32` in cell 3; writes `results_<kind>_<ACCOUNT_TAG>.csv` + `run_manifest.json`; final cell calls `google.colab.files.download`.

- [ ] **Step 1: Write `_template.ipynb`** — 5 cells:

1. **Setup:**
```python
!git clone <REPO_URL> loralink && cd loralink && pip -q install -r loralink_reviewer_response/requirements-colab.txt
%cd loralink
import sys; sys.path.insert(0, ".")
!python loralink_reviewer_response/patch/checksums.py --update && python loralink_reviewer_response/patch/checksums.py --verify
```
2. **Model download:**
```python
from huggingface_hub import snapshot_download
MODEL = "MODEL_PLACEHOLDER"
snapshot_download(MODEL, local_dir=f"./models/{MODEL}",
                  allow_patterns=["*.json","*.txt","*.model","*.safetensors","*.bin",
                                  "merges.txt","vocab.json","tokenizer*"])
```
3. **Params (user edits):**
```python
ACCOUNT_TAG = "acct1"        # unique per Gmail account
SHARD       = "SHARD_PLACEHOLDER"
WALL_BUDGET_MIN = 32
import time; _NB_START = time.time()
def budget_left(): return WALL_BUDGET_MIN*60 - (time.time() - _NB_START)
```
4. **Body (per-notebook):** loop over the notebook's grid, `if budget_left() < PER_RUN_ESTIMATE: break` before each `run_cluster(...)`.
5. **Download:**
```python
import json, glob
from google.colab import files
json.dump({"tag": ACCOUNT_TAG, "shard": SHARD, "done": DONE, "planned": PLANNED,
           "checksums": open("loralink_reviewer_response/patch/SHA256SUMS").read()},
          open(f"run_manifest_{ACCOUNT_TAG}.json","w"), indent=2)
for f in (glob.glob(f"results_*_{ACCOUNT_TAG}.csv")
          + glob.glob(f"results_*_{ACCOUNT_TAG}.summary.csv")
          + [f"run_manifest_{ACCOUNT_TAG}.json"]):
    files.download(f)
# NOTE: main.py writes per-batch rows to results_<kind>_<tag>.csv and the
# single summary row to results_<kind>_<tag>.summary.csv (schemas differ).
# Both must be downloaded and dropped into results/ for aggregate.py.
```

- [ ] **Step 2: Per-notebook bodies** (cell 4 content)

- **00_setup_smoke** — one `run_cluster(n_workers=2, dataset="wikitext", seed=0, model="EleutherAI/gpt-neo-125M", num_samples=6, epochs=1, tag="smoke", results_csv="results_smoke_"+ACCOUNT_TAG+".csv")`; assert the CSV has ≥ 3 loss rows; print PASS/FAIL.
- **01_stat_validation** — `MODEL=gpt-neo-125M`; `SHARD ∈ {"wikitext","e2e"}`; `for seed in range(5): run_cluster(2, SHARD, seed, num_samples=60, epochs=1, compression=True, tag=f"stat-{SHARD}-s{seed}", results_csv=f"results_stat_{ACCOUNT_TAG}.csv")`. PER_RUN_ESTIMATE=200 s.
- **02_task_quality** — `MODEL=microsoft/phi-1_5`; `SHARD` like `"e2e:0"` (dataset:seed); `for arm in ["ON","OFF","reference"]`: `run_cluster(2, ds, seed, model=MODEL, num_samples=50, epochs=1, compression=(arm!="OFF"), eval_holdout=200, tag=f"q-{ds}-s{seed}-{arm}", results_csv=f"results_qsys_{ACCOUNT_TAG}.csv", save_adapters_to=f"adapters/{ds}_s{seed}_{arm}")` — for `reference` use `n_workers=1` and `strategy="smart"` (single-stage pipeline = centralized). Then `evaluate_adapter(MODEL, f"adapters/{ds}_s{seed}_{arm}", ds, arm=arm, seed=seed, out_csv=f"results_quality_{ACCOUNT_TAG}.csv")`. PER_RUN_ESTIMATE=420 s.
- **02b_convergence** — `MODEL=microsoft/phi-1_5`; `run_cluster(2, "e2e", 0, model=MODEL, num_samples=50, epochs=3, compression=True, tag="conv-e2e", results_csv=f"results_converge_{ACCOUNT_TAG}.csv")`. Single run.
- **03_alt_scheduling** — `MODEL=gpt-neo-125M`; `for strat in ["smart","round_robin","proportional","random"]: for seed in range(3): run_cluster(4, "wikitext", seed, strategy=strat, num_samples=30, tag=f"sched-{strat}-s{seed}", results_csv=f"results_sched_{ACCOUNT_TAG}.csv")`. Catch `PartitionInfeasible` → record a row with `n_batches=0, note="infeasible"`. PER_RUN_ESTIMATE=120 s.
- **04_scalability_sim** — `MODEL=gpt-neo-125M`, env `LORALINK_FAKE_BENCHMARK=1`; `for n in [2,3,4,5,6,8]: for rep in range(3): run_cluster(n, "wikitext", 0, num_samples=30, tag=f"scale-n{n}-r{rep}", results_csv=f"results_scale_{ACCOUNT_TAG}.csv")`. Every row's `sim` is `loopback`. PER_RUN_ESTIMATE=180 s.
- **05_network_netem** — `MODEL=gpt-neo-125M`; grid `for delay in [0,25,50,100]: for loss in [0,1,3]: run_cluster(2,"wikitext",0,num_samples=20, netem={"delay_ms":delay,"loss_pct":loss}, tag=f"net-d{delay}-l{loss}", results_csv=f"results_net_{ACCOUNT_TAG}.csv")` (drop the `rate` axis unless budget allows; `SHARD` can carry a delay subset). PER_RUN_ESTIMATE=100 s.

- [ ] **Step 3: Write `build_notebooks.py`** — reads `_template.ipynb`, substitutes `MODEL_PLACEHOLDER` / `SHARD_PLACEHOLDER` / cell-4 body per notebook name, writes the 7 files with `nbformat`.

- [ ] **Step 4: Write the failing test** — `tests/test_notebooks.py`

```python
import nbformat, pathlib, pytest

NBDIR = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
NAMES = ["00_setup_smoke","01_stat_validation","02_task_quality","02b_convergence",
         "03_alt_scheduling","04_scalability_sim","05_network_netem"]

@pytest.mark.parametrize("name", NAMES)
def test_notebook_shape(name):
    nb = nbformat.read(NBDIR / f"{name}.ipynb", as_version=4)
    src = "\n".join(c.source for c in nb.cells)
    assert "WALL_BUDGET_MIN" in src and "budget_left()" in src
    assert "ACCOUNT_TAG" in src
    assert "files.download" in src
    assert "run_cluster(" in src or name == "99_aggregate_report"
    # no run exceeds the ceiling on paper
    assert "num_samples=100" not in src

def test_02_uses_phi_and_others_125m():
    q = nbformat.read(NBDIR/"02_task_quality.ipynb", as_version=4)
    assert "microsoft/phi-1_5" in "\n".join(c.source for c in q.cells)
    s = nbformat.read(NBDIR/"01_stat_validation.ipynb", as_version=4)
    assert "gpt-neo-125M" in "\n".join(c.source for c in s.cells)
```

- [ ] **Step 5: Generate + run tests**

Run: `python loralink_reviewer_response/notebooks/build_notebooks.py && python -m pytest loralink_reviewer_response/tests/test_notebooks.py -v`
Expected: PASS (8 cases)

- [ ] **Step 6: One real end-to-end notebook run** **[colab]**

Execute `00_setup_smoke.ipynb` in Colab Free. Expected: finishes < 12 min, prints `SMOKE PASS`, downloads `results_smoke_*.csv`. Then execute `03_alt_scheduling.ipynb` (fastest full experiment) end-to-end; confirm < 30 min and CSV has rows for all four strategies.

- [ ] **Step 7: Commit**

```bash
git add loralink_reviewer_response/notebooks loralink_reviewer_response/tests/test_notebooks.py
git commit -m "feat: sharded Colab experiment notebooks"
```

---

## Task 13: `99_aggregate_report.ipynb` + `RESPONSE_ABHAY_NIKHIL.md` + `README.md`

**Files:**
- Create: `loralink_reviewer_response/notebooks/99_aggregate_report.ipynb`
- Create: `loralink_reviewer_response/RESPONSE_ABHAY_NIKHIL.md` (template with `{{...}}` placeholders)
- Create: `loralink_reviewer_response/README.md`
- Test: `loralink_reviewer_response/tests/test_response_render.py` **[local]**

**Interfaces:**
- Consumes: `aggregate.build_all`, `aggregate.render_response`, all `results/*.csv`
- Produces: rendered `RESPONSE_ABHAY_NIKHIL.md` with no remaining `{{` placeholders when a full result set is present; `figures/` populated; `README.md` documenting datasets, protocol, metrics, per-number provenance, and the run procedure.

- [ ] **Step 1: Write `RESPONSE_ABHAY_NIKHIL.md` template**

One section per reviewer concern (the 8 from spec §2), each stating: what we did, what we ran `[ours]`, what is `[published, ref N]`, the number(s) via `{{...}}`, and the honest limitation. Include the standing recommendation to soften the cross-platform claim (concern 6). Header block: model tiers, datasets, seeds, "single-box loopback simulation" disclaimer for concerns 5 & 7.

- [ ] **Step 2: Write `99_aggregate_report.ipynb`** — cells:
  1. clone + install
  2. `from google.colab import files; up = files.upload()` → save every uploaded CSV into `results/`
  3. `from loralink_reviewer_response.aggregate import build_all, render_response; s = build_all("results","loralink_reviewer_response/baselines/published_baselines.csv","figures")`
  4. `render_response("figures/summary.json","loralink_reviewer_response/RESPONSE_ABHAY_NIKHIL.md","RESPONSE_ABHAY_NIKHIL.filled.md")`
  5. display every `figures/*.png` inline; `for f in glob("figures/*")+["RESPONSE_ABHAY_NIKHIL.filled.md"]: files.download(f)`

- [ ] **Step 3: Write `README.md`** — sections: Purpose · Model tiers & why · Datasets & splits · Metrics & how computed · Seeds & statistics · What is `[ours]` vs `[published]` · Per-notebook run guide (which account runs which shard) · Reproduce locally · Provenance (checksums, package pins) · Known limitations (125M/1.3B scale, loopback simulation, small n, macOS untested).

- [ ] **Step 4: Write the failing test** — `tests/test_response_render.py`

```python
import json
from loralink_reviewer_response.aggregate import render_response

def test_placeholders_filled(tmp_path):
    sj = tmp_path/"s.json"; sj.write_text(json.dumps(
        {"quality": {"e2e": {"delta_ppl_on_minus_off": 0.4}}}))
    tpl = tmp_path/"tpl.md"; tpl.write_text(
        "PPL delta ON-OFF: {{quality.e2e.delta_ppl_on_minus_off}}")
    out = tmp_path/"out.md"
    render_response(str(sj), str(tpl), str(out))
    assert out.read_text() == "PPL delta ON-OFF: 0.4"

def test_readme_and_template_exist(pkg_dir):
    assert (pkg_dir/"README.md").stat().st_size > 500
    t = (pkg_dir/"RESPONSE_ABHAY_NIKHIL.md").read_text()
    assert t.count("{{") >= 6 and "[published" in t and "loopback" in t.lower()
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest loralink_reviewer_response/tests/test_response_render.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Full-suite run**

Run: `python -m pytest loralink_reviewer_response/tests -v -m "not colab"`
Expected: all non-colab tests PASS.

- [ ] **Step 7: Commit**

```bash
git add loralink_reviewer_response/notebooks/99_aggregate_report.ipynb loralink_reviewer_response/RESPONSE_ABHAY_NIKHIL.md loralink_reviewer_response/README.md loralink_reviewer_response/tests/test_response_render.py
git commit -m "feat: aggregation notebook + reviewer response template + README"
```

---

## Task 14: Package + dry-run verification

**Files:**
- Create: `loralink_reviewer_response/MAKE_ZIP.py` (zips the package minus `results/` contents, `figures/` contents, `models/`, `dataset/`, `__pycache__`)
- Modify: `README.md` (final pin list, final run order)

**Interfaces:**
- Consumes: everything
- Produces: `loralink_reviewer_response.zip` ready to hand to the user; a `VERIFICATION.md` capturing the actual smoke + one-experiment run outputs.

- [ ] **Step 1: Write `MAKE_ZIP.py`** — `shutil.make_archive`, excluding heavy/generated dirs, including a top-level `HOW_TO_RUN.txt` (3 lines: open NB in Colab, set `ACCOUNT_TAG`/`SHARD`, run all; collect CSVs; run 99).

- [ ] **Step 2: Colab dry-run** **[colab]** — in one throwaway account, run `00` then `01` (shard `wikitext`) end to end. Record wall times, paste the `summary` printout and the head of `results_stat_*.csv` into `VERIFICATION.md`. Confirm both < 40 min.

- [ ] **Step 3: Local aggregate dry-run** **[local]** — drop the `01` CSV from Step 2 into `results/`, run `99` logic via `python -c "from loralink_reviewer_response.aggregate import build_all; build_all('loralink_reviewer_response/results','loralink_reviewer_response/baselines/published_baselines.csv','loralink_reviewer_response/figures')"`. Confirm `figures/T1_*` written, `summary.json` has `n=5` for the wikitext shard.

- [ ] **Step 4: Build the zip**

Run: `python loralink_reviewer_response/MAKE_ZIP.py`
Expected: `loralink_reviewer_response.zip` created; unzip in a temp dir and `python -m pytest tests -m "not colab"` still passes.

- [ ] **Step 5: Commit**

```bash
git add loralink_reviewer_response/MAKE_ZIP.py loralink_reviewer_response/VERIFICATION.md loralink_reviewer_response/README.md
git commit -m "chore: package builder + verification record"
```

---

## Self-Review

**1. Spec coverage**

| Spec item | Task(s) |
|---|---|
| §2 concern 1 — statistical validation | 2 (statlib), 6 (metrics), 12 (NB01), 11 (T1) |
| §2 concern 2 — task quality pre/post compression | 4 (toggle), 9 (eval), 12 (NB02), 11 (T2) |
| §2 concern 3 — longer training / convergence | 6 (epochs), 12 (NB02b), 11 (T1 curve) |
| §2 concern 4 — strong baselines (published only) | 10 (baselines), 11 (T3) |
| §2 concern 5 — scalability > 4 nodes (sim) | 3 (strategies not needed) / 4 (fake bench), 8 (cluster), 12 (NB04), 11 (T5) |
| §2 concern 6 — cross-platform (prose only) | 13 (RESPONSE §6) |
| §2 concern 7 — network study | 8 (netem), 12 (NB05), 11 (T6) |
| §2 concern 8 — alternative scheduling | 3 (strategies), 12 (NB03), 11 (T4) |
| §3 model tiers | Global Constraints, 12 Step 2 |
| §4.1 main.py flags/bind/seed/epoch/metrics | 6 |
| §4.2 compression env toggle | 4 |
| §4.3 partitioner strategies | 3 |
| §4.4 data_loader eval split | 5 |
| §4.5 metrics persistence | 1, 6 |
| §4.6 eval_quality.py | 9 |
| §5.1 cluster_launch.py | 8 |
| §5.2 notebook contract + walltime guard | 12 |
| §5.3 published_baselines + SOURCES | 10 |
| §5.4 aggregate outputs T1–T6 + response | 11, 13 |
| §6 honesty rules | 3 (no silent repair), 4 (faithful toggle), 7 (banned-token patch test), 10 (verbatim quotes), 11 (provenance in captions), 12 (`sim` column) |
| §8 risks — NB02 split, netem fallback, multi-bind fallback | 12 (shards), 8 (shim), Global Constraints (ports fallback noted) |
| §9 no git | Task 0, Git note |

No spec section is unassigned.

**2. Placeholder scan** — the only intentional `{{...}}` are in the `RESPONSE_ABHAY_NIKHIL.md` *template* (filled by `render_response`, tested in Task 13). `<FILL>` in Task 10 Step 2 is explicitly gated by a failing test until replaced. `aggregate.py` `_t3`–`_t6` are described with a concrete pattern + required passing tests (Task 11 Step 4) — the executor writes them out; flagged there explicitly. No `TODO`/"handle edge cases"/"similar to Task N" anywhere.

**3. Type consistency**

- `append_rows(csv_path, rows, columns)` — 3-arg form used consistently (Task 1 def, Task 6 calls pass `RUN_COLUMNS`/`SUMMARY_COLUMNS`).
- `run_cluster(...)` keyword signature identical in Task 8 def, Task 8 test, Task 12 notebook bodies (`n_workers`, `dataset`, `seed` positional; rest keyword; `results_csv`, `save_adapters_to`, `tag`, `compression`, `strategy`, `num_samples`, `epochs`, `eval_holdout`, `netem`, `run_timeout_s`, `model`).
- `compute_assignments(strategy, devices, num_layers, layer_size_gb, embedding_size_gb, master_ip, utilization_limit, seed=0)` — same arg order in Task 3 def, tests, and the `partition_model` call site.
- `evaluate_adapter(base_model, adapter_dir, dataset, *, eval_holdout, max_new_tokens, limit, arm, seed, out_csv)` — same in Task 9 def, test, Task 12 NB02.
- `mean_std_ci(values, confidence=0.95) -> {mean,std,ci_lo,ci_hi,n}` — keys used identically in Task 11 (`st["mean"]`, `["n"]`).
- `build_all(results_dir, baselines_csv, out_dir)` / `render_response(summary_json, template_md, out_md)` — consistent Task 11 ↔ Task 13.
- CSV column vocab: `results_stat_*` summary rows come from `SUMMARY_COLUMNS` (Task 1) written by `main.py` (Task 6); `aggregate._t1` reads `mean_loss`, `mean_step_latency_s`, `overall_comp_ratio` — all present in `SUMMARY_COLUMNS`. No mismatch.
- `results_quality_*` columns from `eval_quality._COLS` (Task 9) = `arm,seed,dataset,base_model,perplexity,bleu,rougeL,n_eval,adapter_dir,slice_bounds`; `aggregate._t2` reads `arm,dataset,perplexity,bleu,rougeL` — subset, consistent.

Fixed during review: `run_cluster` gained `model=` and `eval_holdout=` params (were implied by notebooks but missing from the Task 8 signature) — now in the Interfaces block and the implementation. `partition_model` `seed` is sourced from `model_config["seed"]`, set in Task 6 Step 4 — added that line explicitly.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-08-29-loralink-abhay-nikhil-colab.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

**Which approach?**
