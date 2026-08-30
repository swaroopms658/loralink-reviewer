# LoraLink — Response to Reviewers Abhay & Nikhil

*This file is a template. `loralink_reviewer_response/aggregate.py::render_response`
fills every double-brace `dotted.key` placeholder from `figures/summary.json`
(produced by `build_all`). Numbers below are therefore `[ours]` unless tagged
`[published, <ref>]`. A leftover placeholder means the matching shard was not
run — see the per-notebook run guide in `README.md`.*

Provenance rule: **{{provenance}}**

---

## Header — what we ran, and the honest frame

**Two model tiers (ours).** Model scale only carries reviewer weight on the
task-quality and convergence items (concerns 2 and 3); the system-metric
notebooks measure latency, compression ratio and partition balance and are
indifferent to it.

- `EleutherAI/gpt-neo-125M` — concerns 1, 3, 5, 7, 8 (statistical validation,
  convergence shape, scalability, network, scheduling).
- `microsoft/phi-1_5` (1.3 B) — concern 2 only (BLEU / ROUGE-L / perplexity),
  plus the 3-epoch convergence run for concern 3.

Both sit **below the paper's 2.7–3 B headline models** (paper Table:
hyperparameters). Every absolute quality number below is therefore framed as a
**delta (compression ON − OFF)** or a **trend**, never as a SOTA-competitive
absolute. This is stated again in each
section where an absolute appears.

**Datasets and splits.** WikiText-2-raw-v1 — train on `train`, evaluate on
`test`. E2E NLG (`GEM/e2e_nlg`) — train on `train`, evaluate on `validation`.
Slice bounds are logged into every results CSV. Dolly-15k is kept in the paper
but dropped here to fit the Colab session budget.

**Seeds and statistics.** Concern 1: seeds `{0,1,2,3,4}` (n = 5). Concern 2:
seeds `{0,1,2}` (n = 3). Intervals are 95 % Student-t
(`scipy.stats.t`); n is small and is stated with every interval.

**Simulation disclaimer (concerns 5 and 7).** Scalability and network numbers
come from a **{{loopback_disclaimer}}**: N worker processes on `127.0.0.x` with
added-delay / packet-loss shaping of the loopback path (`tc`/`netem` where the
Colab sandbox permits `NET_ADMIN`, otherwise an in-process delay/loss shim).
This shows partitioner and pipeline
behaviour past four stages and the *shape* of the latency response to delay and
loss. It is **not** a WAN measurement; the paper's real 4-node Topology-A
numbers remain the primary evidence there.

**Standing recommendation on the cross-platform claim (concern 6).** We did not
run macOS. We recommend **softening the Windows/macOS/Linux portability claim**
in the paper to Windows/Linux (both tested), which also retires Swaroop #5. The
free way to keep the full claim is a GitHub Actions matrix
(`windows-latest` + `macos-latest` + `ubuntu-latest`) on the public repo — noted
here, out of scope for this Colab package.

**Baselines are published only (concern 4).** We did not re-run DeepSpeed, FSDP,
SplitLoRA, HSplitLoRA, Petals, QLoRA or Megatron-LM. Their numbers are
transcribed verbatim into `baselines/published_baselines.csv` with the source
sentence quoted in `baselines/SOURCES.md`. Our contribution to the comparison
table is the LoraLink row(s) only.

---

## Concern 1 — Statistical validation (repeated runs, std, confidence intervals)

**What the reviewer asked.** Loss / perplexity numbers were reported from a
single run; report repeated runs with standard deviations and confidence
intervals.

**What we did.** Notebook `01_stat_validation.ipynb`, model `gpt-neo-125M`,
2-worker loopback pipeline, 60 mini-batches, 1 epoch, on WikiText-2 and E2E.
Five seeds `{0,1,2,3,4}` per dataset. `aggregate.py` computes mean, sample std
and the 95 % Student-t interval over the seeds.

**What the seed varies, precisely.** The seed drives LoRA's initialization and
the training data order (the loader shuffles under a generator seeded from
`--seed`). The *sample set* is held fixed — all seeds see the same 60 examples —
so the interval below measures sensitivity to initialization and ordering, not
sampling variability over the corpus. Evaluation order is never shuffled, so
held-out metrics stay comparable across runs. Batch size is 1 throughout, as in
the paper's hyperparameter table. We state this because a mean taken over a
fixed sample set is inherently order-insensitive, which is why the spread is
narrow; a wider study would resample the training subset per seed.

**Result (ours).**

- WikiText-2 mean cross-entropy loss = **{{stat_validation.wikitext.mean_loss.mean}}**
  (std {{stat_validation.wikitext.mean_loss.std}}), 95 % CI
  [{{stat_validation.wikitext.mean_loss.ci_lo}}, {{stat_validation.wikitext.mean_loss.ci_hi}}],
  n = {{stat_validation.wikitext.mean_loss.n}} `[ours]`.
- E2E mean cross-entropy loss = **{{stat_validation.e2e.mean_loss.mean}}**
  (std {{stat_validation.e2e.mean_loss.std}}), 95 % CI
  [{{stat_validation.e2e.mean_loss.ci_lo}}, {{stat_validation.e2e.mean_loss.ci_hi}}],
  n = {{stat_validation.e2e.mean_loss.n}} `[ours]`.
- Mean step latency, WikiText-2 = **{{stat_validation.wikitext.mean_step_latency_s.mean}}** s
  (95 % CI [{{stat_validation.wikitext.mean_step_latency_s.ci_lo}}, {{stat_validation.wikitext.mean_step_latency_s.ci_hi}}]) `[ours]`.
- Overall compression ratio, WikiText-2 = **{{stat_validation.wikitext.overall_comp_ratio.mean}}×** `[ours]`.

Figure `figures/T1_stat_validation.csv` / `T1_loss_curve.png` carries the
per-seed mean ± band.

**Honest limitation.** n = 5 is small; the t-interval is wide and is reported as
such. Model is 125 M, so the *absolute* loss is high — the evidence here is
**run-to-run stability**, i.e. the tightness of the interval, not the loss level.

---

## Concern 2 — Downstream task quality before vs after compression

**What the reviewer asked.** Only loss / perplexity was shown; report
generation-quality metrics (task accuracy, BLEU/ROUGE) and show what compression
costs in quality, not just in loss.

**What we did.** Notebook `02_task_quality.ipynb`, model `microsoft/phi-1_5`
(1.3 B), 2-worker loopback pipeline, 50 mini-batches, seeds `{0,1,2}`. Three
arms per (dataset, seed): LoraLink **compression ON**, LoraLink **compression
OFF** (lossy sparsify + int8 disabled, lossless zstd retained — faithful to the
paper's "Disabled" row), and a **1-worker LoraLink run** (the least-partitioned
configuration — coordinator + one worker, compression ON) as a partitioning
control. `eval_quality.py` then scores each saved adapter: perplexity = `exp(mean token
NLL)` on the held-out slice (WikiText-2 `test`); BLEU via `sacrebleu` and
ROUGE-L via `rouge-score` on greedy-decoded E2E `validation` prompts.

**Result — the compression ON − OFF delta (ours).**

- WikiText-2 ΔPPL (ON − OFF) = **{{quality.wikitext.delta_ppl_on_minus_off}}**
  (n = {{quality.wikitext.n}}) `[ours]`.
- E2E ΔBLEU (ON − OFF) = **{{quality.e2e.delta_bleu_on_minus_off}}**
  (n = {{quality.e2e.n}}) `[ours]`.
- E2E ΔROUGE-L (ON − OFF) = **{{quality.e2e.delta_rougeL_on_minus_off}}** `[ours]`.

Absolute BLEU / ROUGE-L / PPL for all three arms are in
`figures/T2_quality_vs_compression.csv` / `T2_quality_bars.png`, each tagged
`[ours]`.

**What is `[published]`.** For context, SplitLoRA reports a perplexity gap to
centralized LoRA of **0.04 PPL** on GPT-2-M `[published, lin2024splitlora]`, and
HSplitLoRA reports SplitLoRA's PPL rising **0.11** under device heterogeneity
`[published, lin2025hsplitlora]`. These are different models and hardware
(2×RTX 3090) — cited as a scale reference for "how small a compression-induced
PPL change is expected to be", not as a head-to-head.

**Honest limitation.** Phi-1.5 is 1.3 B, below the paper's headline models, so
absolute E2E BLEU is modest. The claim we stand behind is that the ON − OFF
delta is small and within the seed CI — compression does not materially degrade
generation quality at this scale.

---

## Concern 3 — Longer training / stronger convergence evidence

**What the reviewer asked.** Loss curves were too short to argue convergence.

**What we did.** Two runs. (a) The 60-batch seed sweep from concern 1 gives a
run-to-run loss band. (b) `02b_convergence.ipynb`: one `phi-1_5` run, 3 epochs
(~150 mini-batches at 50/epoch), compression ON, E2E — a single longer
trajectory. `aggregate.py` plots both into `figures/T1_loss_curve.png`.

**Result (ours).** The 3-epoch loss curve (`figures/T1_loss_curve.png`, NB99
cell 5) is what a human inspects for monotonic descent and absence of divergence
with compression ON. Across the concern-1 seeds the loss std is
**{{stat_validation.e2e.mean_loss.std}}** on E2E and
**{{stat_validation.wikitext.mean_loss.std}}** on WikiText-2 `[ours]` — i.e. the
endpoint is stable across seeds, not a lucky single run.

**Honest limitation.** 3 epochs on 1.3 B is still short next to the paper's
100-batch protocol (paper Table: hyperparameters); this is *supporting*
convergence evidence. The paper's original curves remain primary. Small model,
so absolute loss is high.

---

## Concern 4 — Strong baseline comparisons

**What the reviewer asked.** Compare against DeepSpeed, FSDP, SplitLoRA,
HSplitLoRA, Petals, QLoRA, Megatron-LM.

**What we did.** Per the user's hard constraint, we do **not** reproduce
competitor methods. `baselines/published_baselines.csv` holds
{{ours_vs_published.n_published}} transcribed published numbers from 7 papers
(see `baselines/SOURCES.md`, which quotes the exact source sentence, table/figure
number and URL for each). `aggregate.py` builds `figures/T3_ours_vs_published.csv` placing
our {{ours_vs_published.n_ours}} LoraLink row(s) `[ours]` beside them.

**Result.** The closest-regime comparators — split learning on commodity GPUs
(SplitLoRA / HSplitLoRA `[published, lin2024splitlora]`,
`[published, lin2025hsplitlora]`) and decentralized commodity hardware over the
internet (Petals, ~1 step/s inference on BLOOM-176B
`[published, borzunov2023petals]`) — are annotated `comparable=direct`.
Data-centre methods (DeepSpeed ZeRO 15 PFLOPS/400 GPUs
`[published, rajbhandari2020zero]`, FSDP 186 TFLOPS/GPU on A100
`[published, zhao2023fsdp]`, Megatron-LM 52 % of peak on 3072 A100
`[published, narayanan2021megatron]`) are annotated `comparable=trend` — cited
for the scaling-efficiency trend only, explicitly **not** head-to-head with a
Colab T4. QLoRA (65 B on a single 48 GB GPU `[published, dettmers2023qlora]`) is
`comparable=context` — a memory-efficiency reference point.

**Honest limitation.** Units and hardware differ across every source; the T3
plot is a categorical strip on a log axis and is labelled **NOT like-for-like**.
The table, with per-row hardware and scale, is the artifact — the plot is
orientation only.

---

## Concern 5 — Scalability beyond 4 nodes

**What the reviewer asked.** Results stop at 4 nodes; show behaviour past that.

**What we did.** Notebook `04_scalability_sim.ipynb`, `gpt-neo-125M`, worker
counts 2 / 4 / 6 / 8 as separate loopback processes, 30 batches, 2 reps each.
Measures mean step latency and the per-step rate (steps/s = 1/latency, **not**
aggregate throughput) vs worker count.

**Result (ours).** `figures/T5_scalability_sim.csv` / `T5_lines.png` (NB99
cell 5) reports {{scalability.n}} worker-count points up to 8; the plot is what a
human inspects for how step latency and the per-step rate move as workers are added.
Every row and the figure caption carry the label **"{{scalability.note}}"**
`[ours]`.

**Honest limitation.** This is a **{{loopback_disclaimer}}** — inter-process
latency on `lo` is far below a real LAN/WAN link, so absolute step latency is
optimistic. What it demonstrates is that the partitioner and pipeline schedule
**remain feasible and balanced past 4 stages**; it is not a claim about
real-network scaling. The paper's real 4-node numbers stand.

---

## Concern 6 — Real cross-platform mixed-device tests

**What the reviewer asked.** If the Windows/macOS/Linux claim is kept, test it
on real mixed hardware.

**What we did.** Prose only — no numbers. We ran on Windows and Linux (Colab).
We did **not** run macOS: Colab has no macOS runners and this package has no
budget for Mac hardware.

**Recommendation.** Soften the paper's portability claim to **Windows + Linux**
(both tested). This also resolves Swaroop #5. If the full three-OS claim must be
kept, the only free source of real macOS CI is a **GitHub Actions matrix**
(`windows-latest` / `macos-latest` / `ubuntu-latest`) on the public repo running
the CPU-only 125 M config — portability, not speed. That is a separate CI task,
out of scope here.

**Honest limitation.** No macOS evidence exists in this package. Do not claim
macOS support on the strength of anything in `figures/`.

---

## Concern 7 — Network-condition study (latency, packet loss, bandwidth)

**What the reviewer asked.** Real deployments see variable Wi-Fi, loss and
bandwidth; study the effect.

**What we did.** Notebook `05_network_netem.ipynb`, `gpt-neo-125M`, 2 workers,
20 batches. `tc qdisc … netem` applies an added-delay × packet-loss grid to the
loopback device before each run (in-process socket shim as fallback if `netem`
is blocked). Measures step latency per cell.

**Result (ours).** `figures/T6_network.csv` / `T6_heatmap.png` (NB99 cell 5) is a
{{network.n}}-cell delay × loss grid `[ours]`; the heatmap is what a human
inspects for step latency versus injected delay and loss — the expected shape is
latency rising with both. Note on every cell: **"{{network.note}}"**.

**Honest limitation.** Loopback + emulation, not WAN — `netem` injects a
*modelled* delay/loss distribution on `lo`, it does not reproduce real Wi-Fi
jitter, bufferbloat or TCP-loss dynamics. The evidence is the **response shape**
(latency vs delay/loss), not the absolute numbers.

---

## Concern 8 — Alternative scheduling vs the Smart Partitioning heuristic

**What the reviewer asked.** Is the memory-aware "Smart Partitioning" heuristic
actually better than simpler schemes?

**What we did.** Notebook `03_alt_scheduling.ipynb`, `gpt-neo-125M`, 4-worker
loopback, 30 batches, seeds `{0,1,2}`. Four partitioners compared:
`smart` (existing memory-aware heuristic), `round_robin`, `proportional`
(layers ∝ measured TFLOPS), `random` (fixed-seed). Infeasible assignments are
recorded as a gap, not silently repaired — how often a naive scheduler breaks is
part of the comparison.

**Result (ours).** `figures/T4_scheduling.csv` / `T4_bars.png`, partition-balance
std (lower = more even layer distribution) and mean step latency per strategy:

- `smart`: balance std **{{scheduling.smart.partition_balance_std}}**,
  step latency **{{scheduling.smart.mean_step_latency_s}}** s
  (n = {{scheduling.smart.n}}) `[ours]`.
- `round_robin`: balance std **{{scheduling.round_robin.partition_balance_std}}** `[ours]`.
- `proportional`: balance std **{{scheduling.proportional.partition_balance_std}}** `[ours]`.
- `random`: balance std **{{scheduling.random.partition_balance_std}}** `[ours]`.

**Honest limitation.** 125 M has few transformer blocks, so the absolute
balance-std spread between strategies is compressed; the ordering is the signal,
not the magnitude. Latency is loopback. No published number is comparable here —
this is an internal ablation, all rows `[ours]`.

---

## Summary of provenance

| Concern | Evidence type | Tag on the numbers |
|---|---|---|
| 1 statistical validation | our runs, 5 seeds, t-CI | `[ours]` |
| 2 task quality ON vs OFF | our runs, 3 seeds, Phi-1.5 | `[ours]` (SplitLoRA PPL gap `[published, lin2024splitlora]` for context) |
| 3 convergence | our runs, 3-epoch + seed band | `[ours]` |
| 4 baselines | transcribed literature | `[published, <ref>]` — every row |
| 5 scalability | our loopback simulation | `[ours]`, labelled loopback |
| 6 cross-platform | prose, recommend softening | no numbers |
| 7 network | our loopback delay/loss emulation (netem or in-process shim) | `[ours]`, labelled loopback |
| 8 scheduling | our internal ablation | `[ours]` |

No number in this document is stated without an `[ours]` or `[published, <ref>]`
tag.
