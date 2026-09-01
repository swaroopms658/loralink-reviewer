# LoraLink — reviewer concerns and results

**Date:** 2026-08-31
**Model (ours):** `EleutherAI/gpt-neo-125M` (system metrics), `microsoft/phi-1_5` (task quality)
**Hardware (ours):** Google Colab Free T4, single box, loopback pipeline
**Repo:** https://github.com/swaroopms658/loralink-reviewer

Every number is tagged `[ours]` (we ran it) or `[published, ref]` (transcribed
from a paper, never re-run).

> **Status:** only concern 1 has complete results so far. Four correctness
> defects were found while running this package — the worst disabled attention
> in half the layers — and every measurement taken before they were fixed has
> been discarded. Details in `patch/README.md`. **These defects also affect the
> paper's existing convergence figures and ΔPPL table**; see "Bearing on the
> paper" at the end.

---

## 1 — Statistical validation (repeated runs, std, confidence intervals)

**1 — Reviewer concern:** loss/perplexity numbers were reported from a single
run; report repeated runs with standard deviations and confidence intervals.
(R3.4, R3-Q7)

**2 — Results:** ✅ complete for WikiText-2; E2E shard still to run.

WikiText-2, gpt-neo-125M, 2-worker loopback pipeline, 60 mini-batches, 1 epoch,
compression ON, seeds `{0,1,2,3,4}`.

| statistic | value |
|---|---|
| mean cross-entropy loss | **4.7331** `[ours]` |
| sample std (n=5) | **0.0195** |
| 95 % Student-t CI | **[4.7089, 4.7574]** |
| per-seed means | 4.7267, 4.7153, 4.7629, 4.7189, 4.7420 |
| mean step latency | 0.87 s `[ours]` |
| overall compression ratio | 1.44× `[ours]` |

Run-to-run spread is **±0.4 % relative** — the procedure is reproducible.

*What the seed varies:* LoRA initialization **and** training data order (the
loader shuffles under a generator seeded from `--seed`). The sample set is held
fixed at the same 60 examples, so this interval measures sensitivity to
initialization and ordering, not sampling variability over the corpus. Batch
size is 1 throughout, per the paper's hyperparameter table.

---

## 2 — Downstream task quality, pre/post compression

**1 — Reviewer concern:** report accuracy / generation quality, not only loss and
perplexity, and show the effect of compression on it. 🔴 raised by all three
reviewers.

**2 — Results:** ⚠️ **invalid — must be re-run.** The completed E2E shard was
produced while LoRA was adapting only the MLP on Phi (defect 4 below), so its
BLEU/ROUGE/PPL numbers are not usable. Six shards pending
(`e2e:{0,1,2}`, `wikitext:{0,1,2}`).

What *is* established, from the loss-side ablation on gpt-neo-125M
(30 batches, seed 0, WikiText-2):

| arm | mean loss | interpretation |
|---|---|---|
| 1 worker, lossless | 4.331 `[ours]` | no pipeline hop, no lossy compression |
| 2 workers, lossless | 4.329 `[ours]` | **pipeline hop is free** (−0.002) |
| 2 workers, compressed | 4.367 `[ours]` | **compression costs +0.038 nats (+0.9 %)** |

For reference, `AutoModelForCausalLM` frozen on the identical batches scores
**4.510**; the pipeline coming in below that is the LoRA adapters learning.

---

## 3 — Longer training / stronger convergence

**1 — Reviewer concern:** training runs are too short to demonstrate convergence.

**2 — Results:** ⏳ not yet run (`02b_convergence`, Phi-1.5, 3 epochs, ~150
batches).

Note on interpretation: NB01's single-epoch curve is weak evidence for this
concern regardless, because with batch size 1 and one pass over the data each
step sees an unseen sample, so the curve tracks sample difficulty more than
learning (per-batch spread was 5.07 nats). The 3-epoch run, where samples repeat,
is the real evidence.

---

## 4 — Strong baselines (DeepSpeed, FSDP, SplitLoRA, HSplitLoRA, Petals, QLoRA, Megatron-LM)

**1 — Reviewer concern:** compare against strong distributed / parameter-efficient
baselines. 🔴 raised by all three reviewers.

**2 — Results:** ✅ complete — **published numbers only**, no competitor was
re-run. Full table with per-number hardware, scale and verbatim source quotes in
`baselines/published_baselines.csv` and `baselines/SOURCES.md`.

Direct comparators (same regime — commodity hardware, split/decentralized):

| method | number | source |
|---|---|---|
| SplitLoRA | converges within **0.04 PPL** of centralized LoRA, GPT-2-M on E2E | `[published, lin2024splitlora]` |
| SplitLoRA | centralized LoRA needs **4.8×** the convergence latency | `[published, lin2024splitlora]` |
| HSplitLoRA | **1.5×** convergence speedup vs SplitLoRA, LLaMA-2-7B | `[published, lin2025hsplitlora]` |
| HSplitLoRA | SplitLoRA PPL rises **0.11** under device heterogeneity | `[published, lin2025hsplitlora]` |
| Petals | **1 step/s** generation, BLOOM-176B over the internet | `[published, borzunov2023petals]` |

Context / trend only (data-centre hardware, explicitly **not** like-for-like):
QLoRA (48 GB single-GPU for 65B; 53.1 % MMLU at 4-bit)
`[published, dettmers2023qlora]`; DeepSpeed ZeRO (15 PFLOPS on 400 V100s)
`[published, rajbhandari2020zero]`; PyTorch FSDP (186 TFLOPS/GPU on 128–512
A100s) `[published, zhao2023fsdp]`; Megatron-LM (52 % of peak on 3072 A100s)
`[published, narayanan2021megatron]`.

---

## 5 — Scalability beyond 4 nodes

**1 — Reviewer concern:** results stop at 4 nodes; show behaviour beyond that. 🟠

**2 — Results:** ⏳ not yet run (`04_scalability_sim`, 2/4/6/8 workers).

Will be labelled **single-box loopback simulation, not WAN** in every table and
figure. Concedes that loopback ≠ real network and points to the paper's
real 4-node Topology-A numbers as the physical evidence.

---

## 6 — Real cross-platform mixed-device execution (Windows / macOS / Linux)

**1 — Reviewer concern:** the cross-platform claim is not backed by macOS runs. 🟡

**2 — Results:** ✅ answered in prose — **no runs, and none claimed.**

Recommendation: soften the portability claim to **Windows + Linux verified,
macOS portable by construction** (the stack uses only primitives common to
Windows and POSIX). GitHub Actions `windows/macos/ubuntu-latest` is the only free
source of real macOS runners and would settle it, but it is out of scope for a
Colab package.

---

## 7 — Network-condition study (latency, packet loss, bandwidth)

**1 — Reviewer concern:** no study of how the system behaves under realistic
network conditions. ⚪

**2 — Results:** ✅ complete. 12/12 cells, 0 failed. gpt-neo-125M, 2-worker
loopback pipeline, 20 mini-batches per cell, seed 0.

Mean step latency (s), delay × packet loss:

| | loss 0 % | loss 1 % | loss 3 % | mean |
|---|---|---|---|---|
| **delay 0 ms** | 0.946 | 0.968 | 0.974 | 0.962 |
| **delay 25 ms** | 1.071 | 1.054 | 1.071 | 1.065 |
| **delay 50 ms** | 1.128 | 1.137 | 1.177 | 1.147 |
| **delay 100 ms** | 1.333 | 1.374 | 1.367 | 1.358 |

- **Latency scales linearly with link delay:** 0.962 s → 1.358 s from 0 to
  100 ms, **+41.1 %** `[ours]`.
- **Packet loss is cheap by comparison:** 0 % → 3 % costs **+2.5 %**
  (1.119 → 1.147 s) `[ours]`.
- **Compression ratio is unaffected** by network conditions — 1.390× in all
  twelve cells `[ours]`, as expected since compression happens before transmission.

**The emulation validates itself**, which is the reason these numbers can be
trusted:

- The delay slope implies **3.96 sends per training step**. A 3-stage pipeline
  performs exactly **4** network hops per step (coordinator→w1 and w1→w2 on the
  forward pass, w2→w1 and w1→coordinator on the backward). The measurement
  recovers the topology to within 1 %.
- The loss penalty measured **+0.028 s** against **+0.024 s** predicted by
  4 sends × 3 % × 200 ms RTO.

**Interpretation for the paper:** on a high-latency consumer link, LoraLink's
step time is dominated by **round-trip count**, not by loss. Every additional
pipeline stage adds two hops per step, so deeper pipelines pay a latency cost
that scales with link RTT. That is a stronger argument for the compression
engine than the loss axis: compression reduces bytes per hop, but the hop count
is what latency is actually sensitive to.

**How this was measured (matters for the caption):**

- **Emulation, not WAN.** Single box, loopback, in-process shim. `tc netem` is
  **unavailable on Colab** — it runs under gVisor, which ships the `tc` binary
  but no `sch_netem` module, so every `tc qdisc add` exits 2. Kernel-level
  shaping was not an option; shaping is applied in `NetworkManager.send_message`.
  Baseline (0 ms, 0 %) step latency is 0.946 s, which is loopback, not a
  real link.
- **Delay** is applied per send and is faithful to added latency.
- **Loss** is modelled as a **200 ms retransmission timeout** (Linux
  `TCP_RTO_MIN`), not as a dropped connection. Real packet loss on TCP costs a
  retransmission, not a failed link. The previous model raised `ConnectionError`
  on a simulated drop, which aborted the run and would have measured LoraLink's
  lack of an application-level retry path rather than its behaviour on a lossy
  link.
- **Separately worth stating:** LoraLink has no application-level retry, so a
  genuinely dropped connection *does* end a run. That is a robustness limitation
  worth disclosing on its own, distinct from the latency study above.

---

## 8 — Alternative scheduling vs Smart Partitioning

**1 — Reviewer concern:** Smart Partitioning is not compared against simpler
schedulers. ⚪

**2 — Results:** ⏳ not yet run (`03_alt_scheduling`: smart, round-robin,
compute-proportional, random × 3 seeds).

Naive strategies that produce an infeasible assignment fail loudly rather than
being silently repaired — how often a scheduler breaks is part of the comparison.

---

## Bearing on the paper

Four defects in the model reconstruction meant the framework **could not learn**
until they were fixed. They are in the code path that produced the paper's
convergence figures and ΔPPL ablation.

| # | defect | effect |
|---|---|---|
| 1 | forward pass omitted the final norm, and GPT-Neo's `wpe` | loss ~5322; no learning |
| 2 | loss scored over padding | model learned to emit `<pad>`; loss → 0.22 |
| 3 | `to_empty()` left the causal-mask buffer uninitialized | **attention dead in 6 of 12 GPT-Neo layers**; loss pinned at ~7.4 |
| 4 | stale Phi LoRA target names (`Wqkv`/`out_proj`) | attention never adapted on Phi; 2 modules/layer instead of 6 |

Evidence chain: **5322 → 5.12 → 7.4 → 4.33**, against a frozen HF reference of
**4.510** on identical batches.

**Likely survives, but re-measure:** the 2.4× speedup, 93.6 % latency reduction,
325.97–646.95 MB saved, and Smart Partitioning behaviour — these measure wall
time and bytes moved, which do not depend on the causal mask being correct.
(Sparsification is magnitude-based, so exact ratios may shift.)

**Invalid — produced with attention disabled:** Fig. `loss_e2e` and
Fig. `grid_convergence`; the ΔPPL ablation (0.7625 % on Dolly, −2.09 % vs
−2.02 % on E2E); §542's "does not degrade model convergence"; and the LoRA
Frobenius-norm analysis. The reported ΔPPL values being suspiciously small is
consistent with adapters that barely learned.

**Answer this first:** these defects are verified in the code at `55e1714`. It is
**not confirmed** that this exact state produced the published figures — if they
came from a different branch, the blast radius changes.

Full per-defect write-up with reference-implementation comparisons:
`patch/README.md`. Complete ledger including operational issues:
`VERIFICATION.md`. Offline test suite: 131 passing, up from 76.
