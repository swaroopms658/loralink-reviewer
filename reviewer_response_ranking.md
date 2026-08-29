# LoraLink — Reviewer Comments

Three reviewers, heavily overlapping themes. Below, every comment is grouped, mapped to its source reviewer(s), and assigned.

---

## Swaroop

| # | Fix | Reviewer(s) |
| ---- | --- | ----------- |
| 1 | **Report exact hyperparameters** — batch size, sequence length, learning rate, LoRA rank, sparsity %, quantization bits, network speed, dataset versions, number of runs, random seeds | R2.3, R3-Q3, R3-Q5 |
| 2 | **Add implementation-detail section** — communication protocol, synchronization strategy, memory allocation mechanism, straggler / failure handling | R1.4, R2.3, R3(3) |
| 3 | **Expand Discussion trade-offs** — bandwidth savings vs compression overhead vs latency vs final quality | R1.5, R2.5, R3.7 |
| 4 | **Describe compression mechanism fully** — sparse tensor encoding, index/mask/scale/shape metadata, whether metadata is counted in reported bandwidth, quantization settings, error accumulation | R3(w4), R3-Q4 |
| 5 | **Soften cross-platform claim** — reviewers explicitly offer "reduce this claim" as an option | R2.2, R3(w1), R3-Q1 |

---

## Darshan

| # | Fix | Reviewer(s) |
| ---- | --- | ----------- |
| 6 | **Separate novelty from prior work** — table showing what LoraLink adds vs SplitLoRA / LoRA / pipeline parallelism / gradient compression | R2 (contribution) |
| 7 | **Update arXiv references to peer-reviewed versions** | R2 (references) |
| 8 | **Release code + reproducibility details** | R3-Q8 |
| 9 | **Discuss limitations explicitly** — node failure, TCP loss, privacy, security, device disconnection, fault tolerance | R1.4, R2.5, R3.6 |
| 10 | **Partitioning explanation** — how the heuristic handles activation memory, optimizer memory, sequence length, batch size, pipeline-stage imbalance | R3-Q6, R3(w3) |

---

## Abhay & Nikhil

| Fix | Reviewer(s) |
| --- | ----------- |
| **Statistical validation** — repeated runs, standard deviations, confidence intervals | R3.4, R3-Q7 |
| **Downstream task-quality metrics** — task accuracy / generation quality before and after compression, not only loss / perplexity | R1.3, R2.4, R3(w5) |
| **Longer training / stronger convergence evidence** | R3(w5), R3-Q7 |
| **Strong baseline comparisons** — DeepSpeed, FSDP, SplitLoRA, HSplitLoRA, Petals, QLoRA, Megatron-LM | R1.1, R2.1, R3.1, R3-Q2 (all three reviewers) |
| **Scalability beyond 4 nodes** | R1.2, R2.5, R3.2 |
| **Real cross-platform mixed-device tests** (only if the Win/macOS/Linux claim is kept) | R2.2, R3-Q1 |
| **Network-condition study** — Wi-Fi quality, packet loss, bandwidth variation | R3.5 |
| **Alternative scheduling comparison** vs the Smart Partitioning heuristic | R3.3 |

---

## Abhay & Nikhil — Laptop Feasibility + Reviewer Weightage

**This machine:** i5-10210U (4c/8t), 19.8 GB RAM, MX230 2 GB GPU, 49.6 GB free — but **PyTorch is `2.6.0+cpu`**, so `cuda.is_available()=False` → **all compute is CPU**. Forces small models (`gpt-neo-125M`), single-box "cluster" simulated on localhost.

**Weightage key** (how much a good response moves the review):
🔴 **Critical** = all 3 reviewers / listed as a top weakness · 🟠 **High** = 2 reviewers · 🟡 **Medium** = 1 reviewer, raised twice · ⚪ **Low** = 1 reviewer, single mention.

### ✅ Doable on this laptop

| Item | Verdict | Reviewer basis | Weightage | Requires |
| --- | --- | --- | --- | --- |
| **Statistical validation** (repeated runs, std, CI) | ✅ Full | R3.4, R3-Q7 | 🟡 Medium | Seed as CLI flag; run 5–10 seeds on 125M; `scipy` (installed) for CI |
| **Network-condition study** (latency, loss, bandwidth) | ✅ Full | R3.5 | ⚪ Low | `clumsy` (Win) or WSL2 `tc netem`; no code change — latency already logged |
| **Alternative scheduling** vs Smart Partitioning | ✅ Full | R3.3 | ⚪ Low | Code only — add round-robin / compute-proportional / random partitioners in `device_manager.partition_model` |
| **Downstream task-quality metrics** (before/after compression) | ⚠️ Partial | R1.3, R2.4, R3(w5) | 🔴 **Critical** | `pip install peft evaluate rouge-score`; 125M output weak but the *delta* is real; E2E→BLEU/ROUGE or wikitext→perplexity |
| **Longer training / convergence** | ⚠️ Partial | R3(w5), R3-Q7 | 🟡 Medium | Raise `num_samples` (`main.py:568`) + epoch loop; small model only; overnight CPU runs |

### ❌ Not doable on this laptop (needs external resources)

| Item | Verdict | Reviewer basis | Weightage | Needs instead |
| --- | --- | --- | --- | --- |
| **Strong baselines** (DeepSpeed, FSDP, Megatron, Petals, QLoRA, HSplitLoRA) | ❌ No | R1.1, R2.1, R3.1, R3-Q2 (all 3) | 🔴 **Critical** | Cloud GPU box(es), Linux; QLoRA needs `bitsandbytes`+CUDA |
| **Scalability beyond 4 nodes** (real) | ⚠️ Simulate only | R1.2, R2.5, R3.2 | 🟠 High | ≥5 physical machines / cloud VMs on a LAN (localhost sim ≠ real network) |
| **Cross-platform mixed-device** (Win/macOS/Linux) | ❌ No | R2.2, R3-Q1 | 🟡 Medium | Real macOS + Linux hardware. WSL2 gives Win+Linux only, no macOS. Cheaper: **soften the claim** (Swaroop #5) |

### Priority read

- **Highest impact vs cost:** the two 🔴 Critical items (**baselines**, **quality metrics**) carry the most weight — but baselines need cloud GPU, and quality-metrics is only partial here. Do the cloud baseline runs; do quality-metrics locally on 125M as supporting evidence.
- **Cheap laptop wins that still score:** **statistical validation** 🟡 + **network study** ⚪ + **alt scheduling** ⚪ — all fully doable on CPU, together answer R3.3/R3.4/R3.5/R3-Q7.
- **Unblock first:** `pip install peft evaluate rouge-score`; download `gpt-neo-125M` via `downloader.py`; make seed a CLI flag.

---

## Online / Paid Options — What Each Reviewer Comment They Unblock

Budget is available, so the three ❌ items are no longer blockers. Mapping below: reviewer comment → online resource that solves it.

### Comment → online resource

| Reviewer comment | Weightage | Online option | Config | Est. cost | Status after |
| --- | --- | --- | --- | --- | --- |
| **Strong baselines** — DeepSpeed, FSDP, QLoRA, SplitLoRA, HSplitLoRA, Petals (R1.1, R2.1, R3.1, R3-Q2) | 🔴 Critical | RunPod / Lambda on-demand GPU (free fallback: Kaggle 2×T4) | 1× A100 40GB, or 2× A10 for multi-GPU FSDP/ZeRO | ~$1.2–1.9/h × ~15 h = **$20–30** | ✅ Fully solved. QLoRA's `bitsandbytes` needs CUDA — present here |
| **Scalability beyond 4 nodes** (R1.2, R2.5, R3.2) | 🟠 High | RunPod multi-pod global networking, or GCP `g2-standard-4` / Hetzner VMs in one region | 5 / 6 / 8 nodes, real TCP over real network | 8 × ~$0.5/h × 4 h = **~$16** | ✅ Real network numbers, not localhost sim |
| **Cross-platform mixed-device** Win/macOS/Linux (R2.2, R3-Q1) | 🟡 Medium | GitHub Actions matrix (`windows-latest` + `macos-latest` + `ubuntu-latest`), or Scaleway Mac mini M1 | CPU-only 125M config — claim is portability, not speed | **$0** (Actions, public repo) or ~€0.11/h Scaleway | ✅ Claim can be **kept**, not softened → also retires Swaroop #5 |
| **Longer training / convergence** (R3(w5), R3-Q7) | 🟡 Medium | Same GPU box as baselines, extended run | Larger `num_samples` + epochs; 1.3B model instead of 125M | +$10–20 | ✅ Upgrades laptop's ⚠️ Partial → Full |
| **Downstream task-quality metrics** (R1.3, R2.4, R3(w5)) | 🔴 Critical | Same GPU box — run on a real-size model so outputs aren't toy-grade | BLEU / ROUGE on E2E, perplexity on wikitext | included above | ✅ Upgrades laptop's ⚠️ Partial → Full |
| **Statistical validation** (R3.4, R3-Q7) | 🟡 Medium | Cloud parallel — run seeds concurrently instead of serially on CPU | 5–10 seeds × each config | +$5–10 | ✅ Faster; laptop can also do this free |
| **Release code + reproducibility** (R3-Q8) | — | GitHub public repo + Dockerfile + Zenodo DOI | Same image used for every cloud run | **$0** | ✅ Solves Darshan #8, and makes cloud runs reproducible |
| **Network-condition study** (R3.5) | ⚪ Low | Cloud VMs across regions give *real* latency/loss; `tc netem` on Linux VMs for controlled sweeps | — | negligible | ✅ Stronger than laptop `clumsy` sim |
| **Report exact hyperparameters** (R2.3, R3-Q3, R3-Q5) | — | W&B / MLflow free tier logging every cloud run | — | **$0** | ✅ Auto-captures configs, seeds, versions → Swaroop #1 |
| Alternative scheduling (R3.3), implementation details, trade-offs, compression mechanism, novelty table, limitations, partitioning explanation | ⚪–🟠 | **No online resource needed** — writing / local code only | — | $0 | Unchanged from earlier plan |

### Provider selection

| Provider | Use for | Why |
| --- | --- | --- |
| **RunPod** | Baselines + multi-node scalability | Per-second billing, multi-pod networking, no quota-approval wait |
| **Vast.ai** | Single-node baselines only | Cheapest $/GPU-h, but unreliable interconnect → bad multi-node numbers |
| **GCP / AWS** | Multi-node, if network measurement must be defensible | Same VPC, documented bandwidth — but **GPU quota takes 1–3 days**, request early |
| **Kaggle / Colab** | Free fallback for baselines | 2×T4, ~30 GPU-h/week, $0 — session caps require checkpointing |
| **GitHub Actions** | Cross-platform matrix | Only free source of macOS runners |
| **Scaleway Mac mini M1** | macOS if Actions is too constrained | ~€0.11/h, cheapest real macOS (AWS `mac1` bills 24 h minimum) |

### Total

**~$40–70** clears both 🔴 Critical items plus the 🟠 High one. **~$150–250** additionally buys real-scale models (1.3B–7B) and 3 seeds per config, so reviewers can't dismiss the baselines as toy-scale.

### Rules for spending it

1. **One provider, one instance type, one region** for all reported numbers — mixed hardware across baselines vs scalability will be called out.
2. **Dockerize before paying** — same image on laptop and cloud; also answers R3-Q8.
3. **Freeze the baseline list + configs first** — paid hours should run once, not three times.
4. **Request GPU quota now** if using GCP/AWS.

---
