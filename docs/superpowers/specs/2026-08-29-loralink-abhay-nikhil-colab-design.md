# LoraLink — Abhay & Nikhil Reviewer Response (Colab Free Tier)

**Date:** 2026-08-29
**Status:** Approved design, pre-implementation
**Owner:** raji@msrit.edu

---

## 1. Problem

`reviewer_response_ranking.md` assigns Abhay & Nikhil eight reviewer
concerns. Swaroop and Darshan items are already folded into the paper;
Abhay & Nikhil items are not started. We must produce a **credible,
reproducible reviewer response** that runs inside **Google Colab Free
Tier** (T4 GPU, ~1 h per session; user runs several sessions in
parallel across Gmail accounts).

Hard constraints from the user:

1. Do **not** reproduce baseline/competitor methods. Take their numbers
   from published papers.
2. Run benchmarks **only for our implementation** (LoraLink).
3. Our runs must finish inside the Colab session limit.
4. Do **not** tune or "optimize" the implementation to flatter results.
5. Every reported number is tagged: `[ours]` (we ran it) or
   `[published, ref N]` (taken from literature).
6. Document dataset, protocol, metrics, and the source of each number.

## 2. The eight items and how each is answered

Weightage key from `reviewer_response_ranking.md`:
🔴 Critical (all 3 reviewers) · 🟠 High (2) · 🟡 Medium (1, twice) ·
⚪ Low (1, once).

| # | Item | Weight | Treatment | Artifact |
|---|---|---|---|---|
| 1 | Statistical validation — repeated runs, std, CI | 🟡 | **Run ours.** 5 seeds, report mean ± std and 95 % t-interval. | NB 01 |
| 2 | Downstream task-quality pre/post compression — accuracy / generation quality, not only loss/PPL | 🔴 | **Run ours.** BLEU + ROUGE-L on E2E NLG, perplexity on WikiText-2, for: LoraLink compression ON, LoraLink compression OFF, and a single-process PEFT-LoRA reference. Report the ON→OFF delta. | NB 02 |
| 3 | Longer training / stronger convergence | 🟡 | **Run ours.** 100 mini-batches (paper protocol) loss curves + optional 3-epoch extension on the small model. | NB 01/02 |
| 4 | Strong baselines — DeepSpeed, FSDP, SplitLoRA, HSplitLoRA, Petals, QLoRA, Megatron-LM | 🔴 | **Published numbers only.** Curated table with per-number source. Our contribution to the table is the LoraLink row(s) only. | `baselines/` + NB 99 |
| 5 | Scalability beyond 4 nodes | 🟠 | **Localhost simulation, explicitly labeled.** 2/3/4/5/6/8 worker processes on loopback. Shows partitioner + pipeline behaviour past 4 stages; concedes loopback ≠ WAN; points to the paper's real 4-node Topology-A numbers. | NB 04 |
| 6 | Real cross-platform mixed-device (Win/macOS/Linux) | 🟡 | **Prose only.** Recommend softening the claim (ties Swaroop #5). Note GitHub Actions `windows/macos/ubuntu-latest` matrix as the only free source of real macOS runners, out of scope for this Colab package. | `RESPONSE_ABHAY_NIKHIL.md` |
| 7 | Network-condition study — latency, packet loss, bandwidth | ⚪ | **Run ours.** `tc netem` on the loopback device: delay × loss × rate sweep, measure step latency and throughput. Loopback emulation, labeled as such. | NB 05 |
| 8 | Alternative scheduling vs Smart Partitioning | ⚪ | **Run ours.** Add round-robin / compute-proportional / random partitioners; compare layer-balance and pipeline throughput against the existing memory-aware heuristic. | NB 03 |

Nothing in items 1–3, 5, 7, 8 changes the training maths — only seeds,
sample counts, a compression on/off switch, a partitioner choice, and
metric logging.

## 3. Model, datasets, metrics

- **Model (ours) — two tier.** Model size only earns reviewer weightage
  on the 🔴 quality/convergence items (2, 3); the system-metric
  notebooks are indifferent to it.
  - **`EleutherAI/gpt-neo-125M`** — NB 01, 03, 04, 05 (statistical
    validation, scheduling, scalability, network). These measure loss
    shape, latency, compression ratio, partition balance — none depend
    on generation quality. 125 M keeps every run to 3–5 min.
  - **`microsoft/phi-1_5` (1.3 B)** — NB 02 only (task quality +
    convergence). A 1.3 B LoRA fine-tune produces coherent E2E output,
    so BLEU / ROUGE-L land in a believable range and the compression
    ON→OFF delta is meaningful. Phi-1.5 is already a paper model
    (Table hyperparams). Fits Free T4 (~15 GB) at seq 256, batch 1;
    ~20–35 min per 100-batch pipeline run.
  - 2.7 B / 3 B paper headline models are out — they will not
    pipeline-train inside the Free-T4 hour across seeds.
  - Documented limitation for 125 M results: absolute BLEU/PPL is weak;
    the evidence there is the **delta** and **convergence shape**, not
    absolute quality.
- **Datasets:** WikiText-2 raw (`wikitext/wikitext-2-raw-v1`) for
  perplexity; E2E NLG (`GEM/e2e_nlg`) for BLEU / ROUGE-L. Dolly-15k
  dropped from our runs to fit the session budget (kept in the paper).
- **Split protocol:** train on `train`; evaluate on a disjoint held-out
  slice (`validation` where available, else a fixed tail slice of
  `train` never seen in training). Slice indices are logged.
- **Metrics:**
  - Training: cross-entropy loss per mini-batch (already emitted).
  - Quality: perplexity (`exp(mean NLL)` on held-out), BLEU and
    ROUGE-L via HuggingFace `evaluate`.
  - System: step latency (s), compression ratio, bytes sent, bandwidth
    saved — all already computed inside `pipeline_engine` /
    `compression_engine`; we only persist them.
- **Seeds:** `{0,1,2,3,4}`. Statistics: mean, sample std, 95 %
  confidence interval using Student-t (`scipy.stats.t`).
- **Training length:** 60 mini-batches, 1 epoch for NB 01; 50 for
  NB 02 (Phi-1.5 is slower); 20–30 for NB 03/04/05 (timing only, not
  convergence). Fewer than the paper's 100 to hold the 40-min ceiling;
  the loss trajectory over 50–60 batches is enough to show
  compression does not perturb convergence, and the paper's 100-batch
  curves remain the primary convergence evidence.
- **Convergence-evidence run (item 3):** a separate shard —
  `02b_convergence` — 1 seed, arm ON only, 1 dataset, **3 epochs**
  (~150 batches). ~25 min on Phi-1.5. Run once, not per account.
- **Seeds and sharding.** The two heavy notebooks are sharded across
  accounts; `aggregate.py` merges whatever shards return and states
  the realised `n`.
  - NB 01: seeds `{0,1,2,3,4}`, sharded **by dataset** (WikiText
    shard, E2E shard) → 5 runs/shard.
  - NB 02: sharded **by (dataset × seed)** → 1 shard = 1 dataset × 1
    seed × 3 arms (ON, OFF, reference). Seeds `{0,1,2}` requested →
    up to 6 shards; CI computed on however many come back (≥ 2
    required for an interval, else report per-seed values).
  - NB 03: seeds `{0,1,2}`, single shard.
  - NB 04 / NB 05: seed `{0}`, timing loop repeated 3× internally for
    variance; single shard each (NB 05 optionally sharded by delay).

## 4. Code changes (patch, not rewrite)

All changes are additive and gated by new flags / env vars. Existing
default behaviour is unchanged.

### 4.1 `main.py`

- New argparse flags:
  - `--seed INT` (default `42`) — seed `random`, `numpy`,
    `torch`, `torch.cuda`; set `torch.use_deterministic_algorithms`
    best-effort.
  - `--num-samples INT` (default `100`) — replaces the hard-coded
    `num_samples=100` at the `data_loader.get_data_loader` call
    (`main.py:568`).
  - `--epochs INT` (default `1`) — wrap the `for batch_idx, batch in
    enumerate(train_loader)` loop (`main.py:571`) in an epoch loop;
    global batch counter continues across epochs.
  - `--partition-strategy {smart,round_robin,proportional,random}`
    (default `smart`) — passed into `DeviceManager` /
    `partition_model`.
  - `--run-tag STR` — free-form label written into every metrics row.
  - `--metrics-csv PATH` — append target for metrics (see 4.5).
  - `--host-ip` becomes valid for the `worker` role too (currently
    coordinator-only).
- Bind address: `NetworkManager("0.0.0.0", 29500, …)` at `main.py:445`
  (coordinator) and `main.py:849` (worker) becomes
  `NetworkManager(bind_ip, 29500, …)` where `bind_ip = args.host_ip or
  "0.0.0.0"`. This lets N processes coexist on one box via distinct
  `127.0.0.x` addresses, port 29500 unchanged. The hard-coded `29500`
  in `send_message` calls stays — only the *bind* address changes.
- Seeding happens at the top of `run_coordinator` and `run_worker`
  before any model construction.

### 4.2 `compression_engine.py`

- `OptimizedCompressionEngine.__init__`: read
  `os.environ.get("LORALINK_LOSSY_COMPRESSION", "1")`. When `"0"`,
  override `self.compression_params` so every tensor type has
  `sparsity_ratio = 0.0` and `quantize = False`. Result: gradients and
  activations travel through the lossless zstd wrapper only — no
  sparsification, no int8 quantization. This is the ablation's
  "compression OFF" arm and is faithful to the paper's "Disabled" row
  (lossy transform removed; framing / zstd retained).
- Log the active mode once at init.

### 4.3 `device_manager.py`

- `partition_model(master_ip, strategy="smart")`. Existing body =
  `smart`. Add:
  - `round_robin` — layers dealt out cyclically over healthy devices
    in cluster order, coordinator included.
  - `proportional` — layer count per device ∝ measured TFLOPS
    (`device.stats.flops`), coordinator still capped at 1 (keeps the
    CPU-coordinator invariant), remainder to the fastest worker.
  - `random` — uniform random layer→device assignment with a fixed
    RNG seeded from `--seed`, every device gets ≥ 1 layer.
  - All strategies reuse the existing `PipelineConfig` construction and
    the LM-head / embedding reservation logic; only the per-device
    layer *counts* differ. If a non-smart strategy produces an
    infeasible assignment (device over memory), it fails loudly with
    the same error path as `smart` — we do **not** silently repair,
    because the comparison point is exactly how often naive schedulers
    break.
- Emit the final `{device: layer_count}` map and a balance metric
  (max/mean layer count, std) for NB 03.

### 4.4 `data_loader.py`

- `get_data_loader(..., split="train", eval_holdout=0)`. Add an
  `eval` path: when asked for evaluation data, load the dataset's
  `validation` split if it exists, else take a deterministic tail slice
  of `train` of size `num_samples` offset past anything used in
  training. Log the exact slice bounds.

### 4.5 Metrics persistence

- A small helper (new file `metrics_logger.py`) with
  `append_rows(csv_path, rows: list[dict])`. Header written once.
- Per-batch row: `run_tag, seed, strategy, compression, dataset,
  model, epoch, global_batch, loss, step_latency_s, comp_ratio,
  bytes_sent, bytes_saved, n_workers, timestamp`.
- One summary row per run: totals / means + `wall_time_s`,
  `peak_layers`, `partition_balance_std`.
- Coordinator writes it (it already has `loss_value`, latency, and the
  compression-engine stats object). Hook point: after
  `PIPELINE_ENGINE.backward_step(...)` in the training loop, and once
  more after "Training complete".

### 4.6 Standard evaluation — `eval_quality.py` (new)

Not a reimplementation of LoraLink; a normal HF eval script.

- Load the run's base model (`--base-model`, `microsoft/phi-1_5` for
  NB 02) + the `./lora_adapters` produced by a run via
  `peft.PeftModel.from_pretrained`.
- Perplexity: mean token NLL over the held-out slice → `exp`.
- Generation: greedy decode E2E `meaning_representation` prompts,
  compute BLEU + ROUGE-L against references with `evaluate`.
- Writes one row per adapter into `results_quality_*.csv`.

## 5. Deliverable layout

```
loralink_reviewer_response/
  patch/
    apply_patch.py            applies the diffs below to a fresh repo copy
    *.patch                   unified diffs for §4.1–4.5
  notebooks/
    00_setup_smoke.ipynb
    01_stat_validation.ipynb
    02_task_quality.ipynb
    02b_convergence.ipynb
    03_alt_scheduling.ipynb
    04_scalability_sim.ipynb
    05_network_netem.ipynb
    99_aggregate_report.ipynb
  eval_quality.py
  aggregate.py                CSV -> pandas tables + matplotlib PNG/PDF
  metrics_logger.py
  cluster_launch.py           helper: spawn 1 coordinator + N workers on 127.0.0.x, stream logs, join
  baselines/
    published_baselines.csv
    SOURCES.md
  results/                    user drops downloaded CSVs here (named per account)
  figures/                    output of aggregate.py
  RESPONSE_ABHAY_NIKHIL.md
  README.md
```

### 5.1 `cluster_launch.py`

Single reusable driver imported by every notebook:

```
run_cluster(n_workers, dataset, seed, model="EleutherAI/gpt-neo-125M",
            strategy="smart", compression=True, num_samples=60,
            epochs=1, netem=None, tag="", run_timeout_s=900)
    -> path to metrics csv   (raises TimeoutError past run_timeout_s,
       killing all child processes so one hung run can't eat the budget)
```

- Allocates `127.0.0.1` (coordinator) + `127.0.0.2 … 127.0.0.(n+1)`.
- `subprocess.Popen` per worker (`--role worker --host-ip 127.0.0.x`),
  then the coordinator in-process or as a subprocess.
- Sets `LORALINK_LOSSY_COMPRESSION` env per call.
- Optional `netem`: shell out to `tc qdisc add dev lo root netem …`
  before the run, `tc qdisc del` after (needs Colab root — available).
- Tears down all workers on exit / exception.

### 5.2 Notebook contract

Every experiment notebook:

1. Cell 1 — clone repo + apply patch + `pip install` (peft, evaluate,
   rouge-score, sacrebleu, scipy, zstandard, psutil). ~3–5 min.
2. Cell 2 — download the notebook's model via the repo's
   `downloader.py`: `gpt-neo-125M` (~1 min) for NB 01/03/04/05,
   `microsoft/phi-1_5` (~3–4 min) for NB 02.
3. Cell 3 — parameter block: `ACCOUNT_TAG`, `SHARD` (e.g. `"e2e"` /
   `"wikitext"` / `"wikitext:seed0"`), `WALL_BUDGET_MIN = 32`. The
   only cell the user edits per account.
4. Body — loops calling `cluster_launch.run_cluster(...)`, appending
   to one CSV. **Walltime guard:** before each iteration, if elapsed
   notebook time > `WALL_BUDGET_MIN`, stop the loop, keep what's done.
5. Last cell — write `run_manifest.json`, `files.download(csv_path)`,
   printed summary incl. how many runs completed vs planned.

### 5.2.1 Runtime budget — hard ceiling 40 min per notebook (Free T4)

All figures include the ~5–8 min setup+download. Per-run costs:
125 M / 2-worker / 60 batches ≈ 2–3 min; Phi-1.5 / 2-worker / 50
batches ≈ 5–7 min; 125 M / 30 batches ≈ 1.5 min.

| NB | Model | Shard = 1 account run | Runs | Compute | + setup | Total |
|---|---|---|---|---|---|---|
| 00 smoke | 125 M | whole | 1 (10 batches) | ~3 min | ~6 | **~9** |
| 01 stat-val | 125 M | one dataset | 5 seeds × 60 batches | ~13–15 min | ~6 | **~20** |
| 02 quality | Phi-1.5 | one (dataset, seed) | 3 arms × 50 batches + eval | ~18–22 min | ~8 | **~28–30** |
| 02b converge | Phi-1.5 | whole (run once) | 1 × 3 epochs | ~22–25 min | ~8 | **~33–38** |
| 03 scheduling | 125 M | whole | 4 strat × 3 seeds × 30 batches | ~18 min | ~6 | **~24** |
| 04 scalability | 125 M | whole | 6 sizes × 30 batches × 3 reps | ~22 min | ~6 | **~28** |
| 05 network | 125 M | whole (or by delay) | 12 combos × 20 batches | ~20 min | ~6 | **~26** |
| 99 aggregate | — | whole | no training | ~3 min | ~3 | **~6** |

Every notebook lands under 40 min with margin; the walltime guard makes
overrun impossible — worst case a shard returns partial and
`aggregate.py` uses what it has (and prints what's missing). Each
notebook is independent; the user parallelizes across accounts.

### 5.3 `baselines/published_baselines.csv`

Columns: `method, source_ref, model, params, dataset, metric,
value, hardware, notes`. One row per published number we cite.
`SOURCES.md` gives, per `source_ref`: full citation, arXiv/DOI URL,
the table/section number, and the **verbatim quoted value** so a
reviewer can check it. Numbers to gather (via literature search during
implementation):

- **SplitLoRA** — communication volume / time-to-accuracy on
  GPT-2/LLaMA, from Lin et al. 2024.
- **HSplitLoRA** — perplexity / throughput vs SplitLoRA from its paper.
- **QLoRA** — 4-bit NF4 memory + task accuracy (Dettmers et al. 2023),
  used as the memory-efficiency reference point.
- **DeepSpeed ZeRO / FSDP** — throughput / memory scaling numbers from
  the ZeRO and PyTorch FSDP papers (data-centre GPUs — noted as
  not-comparable-hardware, cited for the scaling trend only).
- **Petals** — decentralized inference/fine-tune throughput over the
  internet (Borzunov et al. 2023) — the closest "commodity hardware
  over WAN" comparator.
- **Megatron-LM** — tensor/pipeline-parallel scaling efficiency
  (Narayanan et al. 2021) — cited for the pipeline-parallel baseline.

Each is clearly annotated with hardware and scale so the comparison is
honest about what is and isn't like-for-like.

### 5.4 `aggregate.py` / NB 99 outputs

- **T1 Statistical validation** — per (dataset, metric): mean ± std,
  95 % CI, n; loss-curve mean ± band plot.
- **T2 Quality vs compression** — BLEU / ROUGE-L / PPL for ON, OFF,
  reference; ΔPPL and ΔBLEU with CI; bar plot with error bars.
- **T3 Our-vs-published** — LoraLink row(s) `[ours]` beside baseline
  rows `[published, ref N]`; separate marker styles in the companion
  scatter (e.g. comm-volume vs quality).
- **T4 Alternative scheduling** — balance std + throughput per
  strategy; grouped bar.
- **T5 Scalability (simulated)** — step latency / throughput vs worker
  count, 2–8; annotated "loopback simulation".
- **T6 Network conditions** — heatmap of step latency over delay ×
  loss; line plot vs bandwidth cap.
- Figures saved PNG + PDF into `figures/`.
- `RESPONSE_ABHAY_NIKHIL.md` templated with the numbers slotted in and
  every claim carrying its `[ours]` / `[published]` tag.

## 6. Honesty and reproducibility rules (enforced in code + prose)

1. Model-scale caveat stated wherever our absolute numbers appear:
   125 M for system-metric notebooks, Phi-1.5 (1.3 B) for quality —
   both below the paper's 2.7–3 B headline models, so quality numbers
   are framed as delta + trend, not as SOTA-competitive absolutes.
2. Loopback runs (NB 04, NB 05) labeled "single-box simulation —
   latency is loopback + emulation, not WAN" in every table/figure
   caption and in the CSV (`n_workers` present, plus a
   `sim=loopback` column).
3. No implementation change touches the training/optimizer/LoRA maths.
   The patch is: flags, bind address, a lossless-only compression
   toggle, extra partitioners, metric logging, eval script.
4. Baseline numbers are transcribed, never re-run; `SOURCES.md`
   carries the verbatim quote + location for each.
5. Seeds, slice indices, package versions, `tc` commands, and the
   patched-source checksums are logged into every results CSV / a
   `run_manifest.json` per notebook.
6. CI is Student-t; n is 5 (NB 01) or 3 (NB 02) — stated with every
   interval, flagged as small.

## 7. Out of scope

- Re-running or re-implementing any competitor method.
- Real multi-machine / multi-region deployment.
- macOS testing (belongs in a GitHub Actions matrix, noted in prose).
- Models larger than 1.3 B for our runs (Phi-1.5 is the ceiling; only
  NB 02 goes above 125 M).
- Changes to LoraLink's algorithms, protocol wire format, or LoRA
  reconstruction.
- Dolly-15k in our runs.

## 8. Risks

| Risk | Mitigation |
|---|---|
| NB 02 (Phi-1.5) overruns 1 h | Default split: E2E on one account, WikiText on another (~30 min each). Eval resumes from saved adapters. Further fallback: drop to 2 seeds, or 60 batches. |
| Phi-1.5 OOM on Free T4 in the pipeline | 2-worker split already halves per-node layers; if still tight, fall back to `EleutherAI/gpt-neo-1.3B` or 3-worker split. Last resort: NB 02 on 125 M with the weightage caveat. |
| `tc netem` blocked in Colab | Fall back to an in-process socket delay/loss shim in `cluster_launch`; label identically. |
| 127.0.0.x multi-bind fails on Colab kernel | Fall back to distinct ports on 127.0.0.1 (needs the `29500` literals parameterized — larger patch, held as plan contingency). |
| Published numbers not truly comparable (hardware/scale) | Every baseline row annotated with hardware + scale; prose frames them as trend/context, not head-to-head, except Petals/SplitLoRA which are the genuine comparators. |
| Phi-1.5 generations still weak on E2E | Report ΔBLEU/ΔROUGE ON-vs-OFF and ΔPPL as primary evidence; absolute scores secondary with the sub-headline-scale caveat. |

## 9. Not using git

This directory is not a git repo. The spec is not committed. The patch
notebooks record the source file checksums instead of a commit hash for
provenance.
