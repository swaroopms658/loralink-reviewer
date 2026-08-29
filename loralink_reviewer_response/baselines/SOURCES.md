# Published baseline sources

Every number in `published_baselines.csv` is transcribed from the paper cited
below, with the verbatim sentence / table cell it came from. Nothing here was
re-run; Colab Free Tier cannot run these methods. These numbers are **not**
like-for-like with LoraLink's Colab-T4 / 125M-1.3B setup - see the `comparable`
and `notes` columns in the CSV.

---

## lin2024splitlora

**Citation:** Zheng Lin, Xuanjie Hu, Yuxin Zhang, Zhe Chen, Zihan Fang, Xianhao Chen,
Ang Li, Praneeth Vepakomma, Yue Gao. "SplitLoRA: A Split Parameter-Efficient
Fine-Tuning Framework for Large Language Models." arXiv preprint arXiv:2407.00952,
2024.

**URL:** https://arxiv.org/abs/2407.00952

**Numbers used:**

- `convergence_latency_ratio_CenLoRA_over_SplitLoRA = 4.8x` and the GPT2-S
  companion values - Section IV-B, Figure 3:

  > The training latency of FedLoRA and CenLoRA for achieving model convergence is approximately 1.7 times and 4.7 times and 2.1 times and 4.8 times that of SplitLoRA on GPT-S and GPT-M, respectively.

  (The CSV row uses the GPT2-M CenLoRA/SplitLoRA ratio, 4.8x.)

- `perplexity_gap_vs_centralized_LoRA = 0.04 PPL` - Section IV-B:

  > SplitLoRA achieves converged accuracy comparable to that of CenLoRA, especially on GPT2-M, where the accuracy difference is less than 0.04.

- Hardware (for the `hardware` column) - Section IV-A, Experimental Setup:

  > The computing capability of each client server is 35.6 TFLOPS (peak performance of one NVIDIA RTX 3090).

**Substitution note:** The task brief also asked for a communication-volume-per-round
figure. SplitLoRA reports communication cost qualitatively / relative to baselines
rather than as one clean transcribable MB-per-round number, so the pinned metrics
above (convergence-time ratio and PPL gap, both from Section IV-B / Figure 3) are
used instead - same paper, same experiment.

---

## lin2025hsplitlora

**Citation:** Zheng Lin, Yuxin Zhang, Zhe Chen, Zihan Fang, Xianhao Chen,
Praneeth Vepakomma, Wei Ni, Jun Luo, Yue Gao. "HSplitLoRA: A Heterogeneous Split
Parameter-Efficient Fine-Tuning Framework for Large Language Models." arXiv
preprint arXiv:2505.02795, 2025.

**URL:** https://arxiv.org/abs/2505.02795

**Numbers used:**

- `convergence_speedup_vs_SplitLoRA_homogeneous = 1.5x` - Section V-A, Figure 20:

  > In the homogeneous setting, HSplitLoRA consistently exhibits the fastest convergence speed, surpassing FT, CenLoRA, HetLoRA, and SplitLoRA by factors of approximately 1.4, 1.3, 1.6 and 1.5 for LLaMA-2-7B, and by factors of 1.5, 1.3, 1.7, and 1.5 for GPT-2-L, respectively.

- `SplitLoRA_PPL_increase_under_device_heterogeneity = 0.11 PPL` - Section V-A,
  Figure 19:

  > In the heterogeneous setting, HetLoRA and SplitLoRA exhibit a significant performance drop, with PPL increasing by approximately 0.38 and 0.1 for LLaMA-2-7B, and 0.38 and 0.11 for GPT-2-L, compared to the homogeneous setting.

  (CSV row uses the GPT-2-L SplitLoRA value, 0.11.)

- Hardware (for the `hardware` column) - Section IV-A:

  > The central server is emulated by an H3C UniServer R5300 G3 server equipped with eight NVIDIA GeForce RTX 3090 GPUs ... we utilize the Jetson AGX Xavier kits, equipped with a 512-core Volta GPU with Tensor Cores.

---

## dettmers2023qlora

**Citation:** Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer.
"QLoRA: Efficient Finetuning of Quantized LLMs." arXiv preprint arXiv:2305.14314,
2023. (Published at NeurIPS 2023.)

**URL:** https://arxiv.org/abs/2305.14314

**Numbers used:**

- `single_GPU_memory_to_finetune = 48 GB` (LLaMA 65B) - Abstract:

  > QLoRA ... reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning task performance.

- `inference_memory_footprint = 6 GB` (Guanaco 7B) and the 13B/33B/65B companions
  - Table 6:

  > Guanaco 7B ... 6 GB; Guanaco 13B ... 10 GB; Guanaco 33B ... 21 GB; Guanaco 65B ... 41 GB

- `accuracy_4bit_NF4_double_quant = 53.1 percent` (MMLU 5-shot, mean over LLaMA
  7B-65B) - Table 4:

  > NF4 + DQ mean 5-shot MMLU accuracy 53.1, vs BFloat16 baseline 53.0 - "4-bit NormalFloat with double quantization matches 16-bit LoRA performance."

**Substitution note:** The exact ">780GB -> <48GB" abstract sentence is used for
the memory row (the brief pointed at a Section 4 table; the abstract states the
same single-GPU 48GB result more cleanly and is the widely cited number). Vicuna
score (99.3% of ChatGPT, abstract) was available but the MMLU Table 4 comparison
was chosen as the quality metric because it directly contrasts 4-bit vs 16-bit.

---

## borzunov2023petals

**Citation:** Alexander Borzunov, Dmitry Baranchuk, Tim Dettmers, Max Ryabinin,
Younes Belkada, Artem Chumachenko, Pavel Samygin, Colin Raffel. "Petals:
Collaborative Inference and Fine-tuning of Large Models." arXiv preprint
arXiv:2209.01188, 2022. (ACL 2023 System Demonstrations.)

**URL:** https://arxiv.org/abs/2209.01188

**Numbers used:**

- `generation_throughput = 1 steps/s` (BLOOM-176B, collaborative inference over
  the internet) - Abstract:

  > we show that this strategy outperforms offloading for very large models, running inference of BLOOM-176B on consumer GPUs with about 1 step per second, which is enough for many interactive LLM applications.

**Substitution note:** The brief asked for a fine-tuning-throughput number; the
paper's cleanly stated, transcribable throughput figure is the ~1 step/s
inference number in the abstract, so that is used (flagged in the CSV `notes` as
inference, not fine-tuning).

---

## rajbhandari2020zero

**Citation:** Samyam Rajbhandari, Jeff Rasley, Olatunji Ruwase, Yuxiong He.
"ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." arXiv
preprint arXiv:1910.02054, 2019/2020. (SC20.)

**URL:** https://arxiv.org/abs/1910.02054

**Numbers used:**

- `aggregate_sustained_throughput = 15 PFLOPS` and
  `max_trainable_params_without_model_parallelism = 13 B` - Abstract:

  > ZeRO can train large models of up to 13B parameters (e.g., larger than Megatron GPT 8.3B and T5 11B) [without requiring model parallelism] ... achieving throughput of 15 Petaflops. This represents an 8x increase in model size and 10x increase in achievable performance ... it trains large models of over 100B parameter with super-linear speedup on 400 GPUs.

---

## zhao2023fsdp

**Citation:** Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang,
Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison,
Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao,
Ajit Mathews, Shen Li. "PyTorch FSDP: Experiences on Scaling Fully Sharded Data
Parallel." arXiv preprint arXiv:2304.11277, 2023. (VLDB 2023.)

**URL:** https://arxiv.org/abs/2304.11277

**Numbers used:**

- `per_GPU_throughput = 186 TFLOPS` (GPT-175B, A100) - Section 5.4, Figure 7(b):

  > With the 175B model, the experiments achieved more than 173 and 186 TFLOPS per GPU with batch size equal to 1 and 2 respectively as shown in Figure 7(b).

- Near-linear scaling context - Section 5.4:

  > Furthermore, the model demonstrated linear scalability from 128 GPUs to 512 GPUs, in terms of TFLOPS.

---

## narayanan2021megatron

**Citation:** Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley,
Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti,
Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, Matei Zaharia. "Efficient
Large-Scale Language Model Training on GPU Clusters Using Megatron-LM." arXiv
preprint arXiv:2104.04473, 2021. (SC21.)

**URL:** https://arxiv.org/abs/2104.04473

**Numbers used:**

- `per_GPU_throughput_percent_of_theoretical_peak = 52 percent` and aggregate
  502 PFLOP/s on 3072 GPUs - Abstract:

  > Our approach allows us to perform training iterations on a model with 1 trillion parameters at 502 petaFLOP/s on 3072 GPUs with achieved per-GPU throughput of 52% of theoretical peak.

- `per_GPU_throughput = 163 TFLOPS` (trillion-parameter model, 3072 A100) -
  Section 5.1 / Table 1 (weak-scaling, 1B to 1T parameters):

  > 163 teraFLOP/s per GPU (52% of theoretical peak) for the trillion-parameter model on 3072 GPUs.
