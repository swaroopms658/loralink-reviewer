# LoraLink — Abhay & Nikhil reviewer-response package (Colab Free Tier)

## Purpose

Produce a credible, reproducible response to the eight reviewer concerns
assigned to Abhay & Nikhil (`reviewer_response_ranking.md`), runnable entirely
inside **Google Colab Free Tier** (T4 GPU, ~1 h/session; the user runs several
sessions in parallel across Gmail accounts). Every reported number is tagged
`[ours]` (we ran it) or `[published, <ref>]` (transcribed from a paper). We do
**not** re-run competitor methods and we do **not** tune LoraLink to flatter
results — the patch is additive flags, a lossless-only compression toggle, extra
partitioners, metric logging, and an eval script.

## Run order

1. **Repo:** `https://github.com/swaroopms658/loralink-reviewer` (public). The
   deliverable zip is the `loralink_reviewer_response/` package only — the
   notebooks `git clone` the full patched repo (cell 1, no edit needed) to get
   `main.py` and the other patched sources.
2. **`00_setup_smoke`** (one throwaway account, Free T4, Runtime → Run all,
   ~9 min) — must print `SMOKE PASS` before you fan out.
3. **`01`–`05` in parallel, one shard per Google account** — download every
   `results_*.csv` + `results_*.summary.csv` each produces:
   - acct1 → `01` `SHARD="wikitext"` · acct2 → `01` `SHARD="e2e"`
   - acct3–5 → `02` `SHARD="e2e:0"` / `"e2e:1"` / `"e2e:2"`
   - acct6–8 → `02` `SHARD="wikitext:0"` / `"wikitext:1"` / `"wikitext:2"`
     (n=3 for the WikiText perplexity delta; skip only if you accept a leftover
     placeholder there)
   - acct9 → `02b` · acct10 → `03` `SHARD=""` · acct11 → `04` `SHARD=""`
     (or `"6,8"` to split) · acct12 → `05` `SHARD=""` (or `"0,25"` to split)
4. **Collect every downloaded CSV into one folder**, open
   `99_aggregate_report.ipynb`, upload them, Run all → `figures/T1..T6.*`,
   `figures/summary.json`, `RESPONSE_ABHAY_NIKHIL.filled.md`. Missing shards
   degrade gracefully (`WARN: no data for T#` + a leftover `{{placeholder}}`).

Quickstart lives in `HOW_TO_RUN.txt` (zip root). A record of the local
packaging/dry-run verification is in `VERIFICATION.md`.

## Model tiers & why

| Tier | Model | Notebooks | Rationale |
|---|---|---|---|
| System metrics | `EleutherAI/gpt-neo-125M` | 01, 03, 04, 05 | Latency, compression ratio, partition balance and loss *shape* do not depend on generation quality. 125 M keeps every run to 3–5 min. |
| Task quality | `microsoft/phi-1_5` (1.3 B) | 02, 02b | A 1.3 B LoRA fine-tune produces coherent E2E output, so BLEU/ROUGE-L land in a believable range and the compression ON→OFF delta is meaningful. Fits Free T4 at seq 256 / batch 1. |

Both tiers are **below the paper's 2.7–3 B headline models** (paper Table:
hyperparameters). Quality numbers are
therefore reported as **deltas and trends**, never as SOTA-competitive
absolutes. This caveat is repeated in `RESPONSE_ABHAY_NIKHIL.md` everywhere an
absolute appears.

## Datasets & splits

| Dataset | HF id | Train | Eval | Used for |
|---|---|---|---|---|
| WikiText-2 raw | `wikitext` / `wikitext-2-raw-v1` | `train` | `test` | perplexity, loss, system metrics |
| E2E NLG | `GEM/e2e_nlg` | `train` | `validation` | BLEU, ROUGE-L |

Dolly-15k is kept in the paper but dropped here to fit the session budget. The
exact eval slice bounds are logged into every results CSV.

## Metrics & how computed

| Metric | Computation |
|---|---|
| Perplexity | `exp(mean token NLL)` over the held-out slice |
| BLEU | `sacrebleu` via HuggingFace `evaluate`, greedy decode |
| ROUGE-L | `rouge-score` via HuggingFace `evaluate` |
| Cross-entropy loss | per mini-batch, as already emitted by `pipeline_engine` |
| Step latency (s) | wall time per optimizer step, logged by the coordinator |
| Compression ratio | bytes-before / bytes-after from `compression_engine` |
| Partition balance std | std of per-device layer count from `device_manager` |

## Seeds & statistics

- Concern 1 (`01_stat_validation`): seeds `{0,1,2,3,4}`, **n = 5**.
- Concern 2 (`02_task_quality`): seeds `{0,1,2}`, **n = 3**.
- Intervals: 95 % **Student-t** (`scipy.stats.t`, `statlib.mean_std_ci`).
- n is small and is printed with every interval. With n < 2, per-seed values are
  reported instead of an interval.

## What is `[ours]` vs `[published]`

- `[ours]` — a number `aggregate.py` computed from a `results_*.csv` this
  package generated on Colab Free T4.
- `[published, <ref>]` — transcribed verbatim from the paper keyed by `<ref>` in
  `baselines/SOURCES.md` (which quotes the exact source sentence + table/figure
  + URL). Never re-run here.
- `render_response` fills `{{...}}` placeholders in `RESPONSE_ABHAY_NIKHIL.md`
  from `figures/summary.json`; all filled values are `[ours]`. Published numbers
  are written literally into the template with their `[published, <ref>]` tag.

## Per-notebook run guide

Each notebook is independent. Assign one shard per Gmail account, run in
parallel, download the `results_*` CSVs, drop them into `results/`, then run
`99_aggregate_report.ipynb` once.

| Notebook | Model | Answers | `SHARD` value | ~time | Outputs (download all) |
|---|---|---|---|---|---|
| `00_setup_smoke` | 125 M | env smoke test | `""` | ~9 min | `results_smoke_*.csv` |
| `01_stat_validation` | 125 M | concerns 1, 3 | `"wikitext"` **or** `"e2e"` (one dataset per account) | ~20 min | `results_stat_*.csv` + `.summary.csv` |
| `02_task_quality` | Phi-1.5 | concern 2 | `"e2e:0"`, `"e2e:1"`, `"e2e:2"`, `"wikitext:0"`, `"wikitext:1"`, `"wikitext:2"` (one `dataset:seed` per account) | ~28–30 min | `results_quality_*.csv` (aggregated); `results_qsys_*.csv` (retained as raw Phi-1.5 per-batch training log for the appendix, not aggregated) |
| `02b_convergence` | Phi-1.5 | concern 3 | — (n/a, single hardcoded run) | ~33–38 min | `results_converge_*.csv` |
| `03_alt_scheduling` | 125 M | concern 8 | `""` (whole notebook, one account) | ~24 min | `results_sched_*.csv` + `.summary.csv` |
| `04_scalability_sim` | 125 M | concern 5 | `""` for the full 2/4/6/8 sweep, or `"6,8"` to split across accounts | ~28 min | `results_scale_*.csv` + `.summary.csv` |
| `05_network_netem` | 125 M | concern 7 | `""` for all delays, or `"0,25"` to split by delay | ~26 min | `results_net_*.csv` + `.summary.csv` |
| `99_aggregate_report` | — | merges everything | — | ~6 min | `figures/*`, `RESPONSE_ABHAY_NIKHIL.filled.md` |

Suggested account map: acct1 → `01` wikitext · acct2 → `01` e2e ·
acct3–5 → `02` `e2e:{0,1,2}` · acct6–8 → `02` `wikitext:{0,1,2}` ·
acct9 → `02b` · acct10 → `03` · acct11 → `04` · acct12 → `05`.
The `02` `wikitext:*` shards give n=3 for the WikiText perplexity delta in
Concern 2 — skip them only if you accept a leftover placeholder there.
`aggregate.py` tolerates missing shards — it prints `WARN: no data for T#` and a
leftover `{{placeholder}}` for whatever did not come back.

The `99` notebook also runs locally: `google.colab` imports are guarded, so
`jupyter nbconvert --execute` against a hand-populated `results/` runs cells 3–4
(aggregate + render). Notebooks 00–05 are generated —
`python -m loralink_reviewer_response.notebooks.build_notebooks` rebuilds them
from `_template.ipynb`; `99` is hand-maintained.

## How to run locally

```bash
# offline test suite (no GPU / network / live cluster)
pytest -q -m "not colab"

# regenerate notebooks 00–05 from the template
python -m loralink_reviewer_response.notebooks.build_notebooks

# dry-run the aggregation path against a populated results/
jupyter nbconvert --to notebook --execute \
  loralink_reviewer_response/notebooks/99_aggregate_report.ipynb
```

## Provenance

- Source integrity: `patch/SHA256SUMS` records the checksum of every patched
  source file. Notebook cell 1 runs `patch/checksums.py --verify` (bare, no
  `--update`) — it checks the cloned tree against the committed `SHA256SUMS` and
  aborts the run on any mismatch.
- Dependency pins: `requirements-colab.txt` installs **only what Colab Free
  lacks** — `datasets` 2.21.0 (the 3.x line dropped the script-based loader
  GEM/e2e_nlg needs), `peft` 0.12.0, `evaluate` 0.4.2, `rouge-score` 0.1.2,
  `sacrebleu` 2.4.3, `zstandard` 0.23.0. Colab's own `torch`, `transformers`,
  `accelerate`, `scipy`, `pandas`, `matplotlib`, `numpy`, `psutil` are left as
  shipped; pinning/downgrading them forced pip into a ~20-min resolver backtrack
  against Colab's CUDA torch build (observed on a live run).
- Branch: `reviewer-response-abhay-nikhil`.
- Each notebook writes a `run_manifest_*.json` with the account tag, shard,
  runs-completed-vs-planned, and the `SHA256SUMS` contents.

## Known limitations

- **Model scale.** 125 M (system) / 1.3 B (quality), both below the paper's
  2.7–3 B headline models. Absolute quality numbers are weak; the evidence is the
  compression ON−OFF **delta** and the convergence **shape**.
- **Loopback simulation (concerns 5 & 7).** Scalability and network numbers are a
  single-box loopback simulation with added-delay / packet-loss shaping
  (`tc`/`netem` where the sandbox allows it, else an in-process shim) — **not WAN**.
  Absolute latency is optimistic; only the response *shape* is claimed.
- **Small n.** n = 5 (concern 1) / n = 3 (concern 2); Student-t intervals are
  wide and flagged as such.
- **macOS untested (concern 6).** No macOS runs. Recommendation: soften the
  portability claim to Windows + Linux, or add a GitHub Actions matrix.
- **Colab session caps.** ~1 h/session forces sharding across accounts; a
  walltime guard in each notebook stops mid-loop and keeps partial results.

## Note on `pytest.ini`

The repo ships a minimal `pytest.ini` (just the `colab` marker). Its only job is
to be the config anchor: pytest stops directory discovery at the file that
defines `[pytest]`, which shields the suite from an unrelated `setup.cfg` in the
dev machine's home directory that would otherwise be picked up as pytest config.
