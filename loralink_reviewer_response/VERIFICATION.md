# VERIFICATION — LoraLink reviewer-response package (Abhay & Nikhil)

What was actually run on the packaging machine (Task 14), the exact commands, and
their output. The Colab GPU work that **could not** be run here is listed at the
bottom.

- Repo: `reviewer-response-abhay-nikhil` @ `5db8421`, clean tree, no remote.
- Python 3.12.7, Windows 11. (Colab Free ships 3.10/3.11 — see caveats.)

---

## 1. Offline test suite (full repo)

```
python -m pytest loralink_reviewer_response -q -m "not colab"
```

```
74 passed, 2 deselected in ~50s
```

The 2 deselected are the `@pytest.mark.colab` tests (see bottom).

---

## 2. Local end-to-end aggregation dry-run

The Colab GPU notebooks cannot run here, so a schema-faithful set of **fake**
result CSVs was hand-written into a temp `results/` dir using the real schemas
(`metrics_logger.RUN_COLUMNS` / `SUMMARY_COLUMNS`, `eval_quality._COLS`) and the
aggregation pipeline was run against them. Numbers below are synthetic — only the
schema, file wiring, and placeholder coverage are being verified.

Fake inputs written (temp `results/`):

| file | schema | shape |
|---|---|---|
| `results_stat_wikitext.summary.csv` | `SUMMARY_COLUMNS` | 5 seeds, `dataset=wikitext` |
| `results_stat_wikitext.csv` | `RUN_COLUMNS` | 5 seeds × 20 per-batch rows |
| `results_stat_e2e.summary.csv` | `SUMMARY_COLUMNS` | 5 seeds, `dataset=e2e` |
| `results_stat_e2e.csv` | `RUN_COLUMNS` | 5 seeds × 20 per-batch rows |
| `results_converge_phi.csv` | `RUN_COLUMNS` | 2 epochs × 6 batches, per-batch loss |
| `results_quality_phi.csv` | `eval_quality._COLS` | arms ON/OFF/reference × wikitext+e2e × 3 seeds |
| `results_sched_x.summary.csv` | `SUMMARY_COLUMNS` + `note` | `smart`, `round_robin`, `random`, `proportional` (`note=infeasible`, `n_batches=0`) |
| `results_scale_x.summary.csv` | `SUMMARY_COLUMNS` | `n_workers` = 2, 4, 6, 8 |
| `results_net_x.summary.csv` | `SUMMARY_COLUMNS` + `delay`,`loss` | 4 delays × 3 loss rates = 12 cells |

Command (equivalent to notebook `99` cells 3–4):

```python
from loralink_reviewer_response.aggregate import build_all, render_response
s = build_all(<tmp>/results,
              "loralink_reviewer_response/baselines/published_baselines.csv",
              <tmp>/figures)
render_response(<tmp>/figures/summary.json,
                "loralink_reviewer_response/RESPONSE_ABHAY_NIKHIL.md",
                <tmp>/RESPONSE.filled.md)
```

### Output — `figures/` produced

```
summary.json
T1_stat_validation.csv   T1_loss_curve.png   T1_loss_curve.pdf
T2_quality_vs_compression.csv   T2_quality_bars.png   T2_quality_bars.pdf
T3_ours_vs_published.csv   T3_scatter.png   T3_scatter.pdf
T4_scheduling.csv   T4_bars.png   T4_bars.pdf
T5_scalability_sim.csv   T5_lines.png   T5_lines.pdf
T6_network.csv   T6_heatmap.png   T6_heatmap.pdf
```

All 6 `T*.csv` + `T*.png` + `T*.pdf` present. `build_all` printed no `WARN:` lines.

### `summary.json`

`json.load()` succeeds. Top-level keys:
`loopback_disclaimer, network, ours_vs_published, provenance, quality,
scalability, scheduling, stat_validation`.

`T4_scheduling.csv` (infeasible strategy shown as an empty-value / `infeasible`
row, not a crash):

```
strategy,partition_balance_std,mean_step_latency_s,status,n,tag
proportional,,,infeasible,0,[ours]
random,0.71,0.52,ok,1,[ours]
round_robin,0.55,0.47,ok,1,[ours]
smart,0.18,0.4,ok,1,[ours]
```

### `RESPONSE.filled.md` — leftover `{{...}}` placeholders

**0 leftover placeholders.** All 36 template placeholders were filled from
`summary.json`.

One cosmetic note: `proportional` is the fake `infeasible` strategy, so
`{{scheduling.proportional.partition_balance_std}}` renders as the literal
`None` ("- `proportional`: balance std **None** `[ours]`."). That is the intended
"infeasible → no value" display, not an unfilled placeholder; with real data a
strategy that IS feasible renders a number, and a genuinely infeasible one still
renders `None` here by design.

---

## 3. Notebook builder is idempotent

```
python loralink_reviewer_response/notebooks/build_notebooks.py
git status --porcelain
```

`build_notebooks.py` rewrote the 7 sharded notebooks; `git status` showed **no
change to any tracked file** (only the new untracked Task-14 files). Idempotent.

---

## 4. `MAKE_ZIP.py` + packaged-copy tests

```
python loralink_reviewer_response/MAKE_ZIP.py
```

```
wrote .../loralink_reviewer_response.zip
  size:  ~79 KB
  files: 52
```

`zipfile.namelist()` spot-check: contains `loralink_reviewer_response/aggregate.py`,
`.../README.md`, `.../HOW_TO_RUN.txt`, `HOW_TO_RUN.txt` (zip root copy),
`.../VERIFICATION.md`, `.../notebooks/99_aggregate_report.ipynb`,
`.../patch/SHA256SUMS`, `.../results/.gitkeep`, `.../figures/.gitkeep`,
`.../pytest.ini` (synthesized config anchor). Contains **no** `__pycache__/`,
`*.pyc`, `.pytest_cache`, `adapters/`, `*.zip`, and no `results/*.csv` /
`figures/*.png`.

### Packaged copy dropped back into the repo it ships alongside

`git archive HEAD` → temp dir, delete `loralink_reviewer_response/`, unzip the
package in its place, then:

```
python -m pytest loralink_reviewer_response -q -m "not colab"
```

```
74 passed, 2 deselected
```

This is the meaningful standalone check: the packaged files are complete and
correct for their target environment (the full patched repo that notebook cell 1
`git clone`s).

### Package unzipped completely on its own (no surrounding repo)

```
cd <unzip dir>
python -m pytest loralink_reviewer_response/tests -q -m "not colab" --continue-on-collection-errors
```

```
50 passed, 2 deselected, 11 failed, 2 errors
```

The 13 non-passing tests **all** require repo-root source files that are
deliberately NOT in this package (`main.py`, `compression_engine.py`,
`benchmarking.py`, `device_manager.py`, `data_loader.py` — the notebooks obtain
them via `git clone` of the full patched repo):

| test | needs |
|---|---|
| `test_cluster_smoke.py::test_ip_plan_and_signature` | `<repo>/main.py` |
| `test_cluster_smoke.py::test_child_cmds_use_absolute_main` | `<repo>/main.py` |
| `test_compression_toggle.py` (3 tests) | `compression_engine`, `benchmarking` |
| `test_compute_assignments.py` (collection error) | `device_manager` |
| `test_data_holdout.py` (collection error) | `data_loader` |
| `test_main_cli.py` (5 tests) | `main.py` |
| `test_patch_roundtrip.py::test_checksums_verify_clean` | checksums of the 5 root sources |

Nothing here indicates a packaging defect — it is the direct consequence of the
"package only, notebooks clone the patched repo" design (HOW_TO_RUN.txt step 1).

---

## 5. Dependency pins (`requirements-colab.txt`)

```
python -m pip install -r loralink_reviewer_response/requirements-colab.txt --dry-run
```

Resolved cleanly (exit 0) on Python 3.12 — no version bumps required. Pins are
left **as specified** (transformers 4.44.2, datasets 2.21.0, accelerate 0.34.2,
peft 0.12.0, evaluate 0.4.2, rouge-score 0.1.2, sacrebleu 2.4.3, scipy 1.13.1,
pandas 2.2.2, matplotlib 3.9.2, zstandard 0.23.0, psutil 6.0.0). A full install +
import was **not** exercised against an actual Colab runtime (Python 3.10/3.11 +
preinstalled CUDA stack); the pins are the versions known-good on Colab Free at
authoring time and are installed unchanged by notebook cell 1.

---

## NOT verified here — needs a Colab GPU session

- **`@pytest.mark.colab` tests** (2, deselected in every run above):
  - `loralink_reviewer_response/tests/test_cluster_smoke.py::test_two_worker_run_writes_rows`
    — spawns a real 2-worker loopback cluster via `run_cluster`.
  - `loralink_reviewer_response/tests/test_eval_quality.py::test_eval_runs_on_base_only`
    — loads a model and runs `evaluate_adapter`.
- **Notebooks 00–05 end-to-end on Free T4:** `00_setup_smoke`, `01_stat_validation`,
  `02_task_quality`, `02b_convergence`, `03_alt_scheduling`, `04_scalability_sim`,
  `05_network_netem` — none has been executed. Wall-time budgets (~9–38 min each,
  per README) and the walltime guard are unmeasured. Real `results_*.csv` /
  `results_*.summary.csv` schemas were modelled from `metrics_logger` /
  `eval_quality`, not observed.
- **`99_aggregate_report.ipynb` executed as a notebook** (`nbconvert --execute`):
  only its cell 3–4 logic (`build_all` + `render_response`) was run directly, not
  the notebook itself.
- **Full `pip install` against a real Colab runtime** (see §5).
- **The `<SET_REPO_URL>` clone flow** — depends on the user pushing the patched
  repo to a remote, which does not exist yet.
