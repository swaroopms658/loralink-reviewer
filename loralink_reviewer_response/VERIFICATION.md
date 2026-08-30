# VERIFICATION — LoraLink reviewer-response package (Abhay & Nikhil)

What was actually run on the packaging machine (Task 14), the exact commands, and
their output. The Colab GPU work that **could not** be run here is listed at the
bottom.

- Repo: `reviewer-response-abhay-nikhil`, clean tree, no remote. Dry-run first
  captured at `e66cd00`; test counts below refreshed after the final-review fix
  commit `29a89a3` (76 offline tests).
- Python 3.12.7, Windows 11. (Colab Free ships 3.10/3.11 — see caveats.)

---

## 1. Offline test suite (full repo)

```
python -m pytest loralink_reviewer_response -q -m "not colab"
```

```
76 passed, 2 deselected in ~50s
```

The 2 deselected are the `@pytest.mark.colab` tests (see bottom).

---

## 2. Local end-to-end aggregation dry-run

The Colab GPU notebooks cannot run here, so a schema-faithful set of **fake**
result CSVs was hand-written into a temp `results/` dir using the real schemas
(`metrics_logger.RUN_COLUMNS` / `SUMMARY_COLUMNS`, `eval_quality._COLS`) and the
aggregation pipeline was run against them. Numbers below are synthetic — only the
schema, file wiring, and placeholder coverage are being verified.

> **Fake-schema caveat.** These hand-written CSVs predate the final-review fixes
> and do **not** track every real-run schema detail — e.g. the fake
> `results_sched_*.summary.csv` carries a `note` column only for genuinely
> infeasible strategies, and the fake `results_net_*` carries `delay`/`loss`
> columns that real runs derive from the `run_tag` instead. The real aggregation
> code paths (`_t4`..`_t6`, including the ragged-header guard and the
> `run_tag`-parsed delay/loss) are exercised with faithful fixtures in
> `loralink_reviewer_response/tests/test_aggregate.py`; treat that suite, not
> these fakes, as the schema authority.

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
76 passed, 2 deselected
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

**Superseded.** The original full pin set (transformers 4.44.2, scipy 1.13.1,
pandas 2.2.2, matplotlib 3.9.2, accelerate 0.34.2, psutil 6.0.0, …) resolved
cleanly under `--dry-run` on Python 3.12 but on a **live Colab Free run stalled
pip for 20+ min** — downgrading Colab's preinstalled CUDA `torch` stack sent the
resolver into a backtrack. `requirements-colab.txt` was cut down to only the
packages Colab lacks: `datasets==2.21.0`, `peft==0.12.0`, `evaluate==0.4.2`,
`rouge-score==0.1.2`, `sacrebleu==2.4.3`, `zstandard==0.23.0`. Colab's own
`torch`/`transformers`/`accelerate`/`scipy`/`pandas`/`matplotlib`/`numpy`/`psutil`
are used as shipped. A full install + import + run against a live Colab runtime
is still pending confirmation.

---

## 6. Live Colab run — what it found

The package was executed on a real Colab Free T4 (2026-08-30). Four defects
surfaced that no offline test could have caught; all four are fixed.

| # | Symptom on Colab | Cause | Fix |
|---|---|---|---|
| 1 | `pip install` stalled 20+ min | pinning `torch`/`transformers`/`scipy`/… downgraded Colab's preinstalled CUDA stack, sending pip's resolver into a backtrack | `requirements-colab.txt` cut to only what Colab lacks |
| 2 | coordinator exited 1 after 30 s, "Failed to get results from `['127.0.0.2','127.0.0.3']`" | on loopback every `127.0.0.x` client's connection is seen by `accept()` as `127.0.0.1`, so all workers collapsed to one identity | workers stamp `metadata["src_ip"]`; coordinator prefers it over the socket peer |
| 3 | `HfUriError: Repository id must be 'namespace/name'` | newer `huggingface_hub` rejects the bare `wikitext` id | `Salesforce/wikitext` |
| 4 | **loss 2 000–11 000, no convergence** | forward pass omitted the final norm and GPT-Neo's `wpe` | see `patch/README.md` §2 |
| 5 | after fixing 4: loss fell 10.29 → **0.22** in 60 steps | `cross_entropy` scored the padding `data_loader` adds to 256 tokens; model learned "emit padding" | `build_masked_labels` + `ignore_index=-100`, `patch/README.md` §2b |
| 6 | after fixing 5: loss stuck at **~7.4**, flat, identical with compression on/off | `to_empty()` left GPT-Neo's causal-mask buffer uninitialized (non-persistent ⇒ absent from the checkpoint); it materialized all-`False`, so attention masked everything and the model predicted from token statistics alone | `build_reference_block` + `restore_nonpersistent_buffers`, `patch/README.md` §2c |
| 7 | cross-seed std 0.0016 — seeds varied almost nothing | loader used `shuffle=False`, so every seed saw identical data in identical order; only LoRA's `A` init differed | seeded training shuffle, `patch/README.md` §1b |
| 8 | NB02 arm 2 died at batch 1 after arm 1 succeeded | `evaluate_adapter` loaded a full fp32 Phi-1.5 (~5.3 GB) on the GPU in the notebook process between arms and never released it | `_release_model()` in a `finally` |
| 9 | that failure surfaced only as a bare 300 s gradient timeout | worker stdout/stderr was inherited and lost, so a worker dying mid-run left no evidence | per-worker log files; tails attached to the raised error |

Defect 4 is the significant one. Evidence that isolated it, both arms 25 batches
on gpt-neo-125M:

```
compression OFF   first 5: [3895, 9150, 5006, 4479, 4008]   mean 6442.70
compression ON    first 5: [5479, 8918, 5410, 5922, 3162]   mean 6591.78
```

Identical between arms ⇒ not the compression engine. Batch 0 of a *pretrained*
model at 5 479 (correct ≈ 4–6, uniform-random ≈ ln 50257 ≈ 10.8) ⇒ logits
inflated ~500×, i.e. a missing normalization, confirmed against
`transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoModel.forward`.

Three 5-seed NB01 shards were run during this investigation — before any fix
(loss ~5 300), after the norm fix (10.29 → 0.22, scoring padding), and after the
padding fix (~7.4, flat, attention dead). **All three are discarded.** Every
result CSV must be regenerated on the fixed code.

The control that isolated defect 6 — three arms, 30 batches each:

```
1 worker,  lossless    mean 7.604   first 8.779
2 workers, lossless    mean 7.321   first 7.703
2 workers, compressed  mean 7.533   first 7.911
```

A lossless single-worker run being no better than a compressed two-worker run is
what ruled out both compression and the pipeline hop, and pointed at the model
reconstruction itself.

### Closing check — reference forward vs the pipeline

The check that should have existed from the start: run
`AutoModelForCausalLM` over the same batches from the same `get_data_loader`
call and compare. On 30 wikitext batches, gpt-neo-125M, frozen:

```
HF AutoModelForCausalLM reference        4.510
```

Same batches through the fixed pipeline (mean over 30 training batches, so LoRA
is adapting and a small improvement on the frozen reference is expected):

```
1 worker,  lossless    4.331   (-0.179 vs reference)
2 workers, lossless    4.329   (-0.181)
2 workers, compressed  4.367   (-0.143)
```

The pipeline now reproduces the reference. Two results follow, and both are
meaningful only because of that:

- **the pipeline hop is free** — 1 worker 4.331 vs 2 workers 4.329 (-0.002);
- **lossy compression costs +0.038 nats (+0.9 %)** — 4.329 → 4.367.

Before the fixes all three arms sat at 7.3–7.6 regardless of configuration,
because attention was disabled in every one of them: the ablation was measuring
nothing at all.

Offline suite after all three fixes: **105 passed, 2 deselected** (was 76 + 2).
New: `tests/test_final_norm.py`, `tests/test_loss_masking.py`,
`tests/test_nonpersistent_buffers.py`.

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
- **`04_scalability_sim` (n=8) and `05_network_netem` are not merely unmeasured
  but *likely to hit partial failure* on a real Free T4** — NB04 can OOM at n=8
  (9 model-loading processes on one T4) and NB05's packet-loss cells can abort
  when a dropped send raises `ConnectionError` (P(abort) ≈ 45 % at `loss_pct=1`,
  ≈ 84 % at 3 %). The per-cell `try/except` added in the final review degrades
  these to *partial* results (the failed cells are skipped and printed) instead
  of losing the whole sweep.
- **`99_aggregate_report.ipynb` executed as a notebook** (`nbconvert --execute`):
  only its cell 3–4 logic (`build_all` + `render_response`) was run directly, not
  the notebook itself.
- **Full `pip install` against a real Colab runtime** (see §5).
- **The clone flow** — the repo is pushed
  (`github.com/swaroopms658/loralink-reviewer`, public, master); cell 1 of every
  notebook clones it directly. Not yet exercised against a live Colab runtime.
