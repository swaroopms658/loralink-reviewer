# LoraLink — correctness findings that affect the paper

**Date:** 2026-08-31
**Found by:** running the Abhay & Nikhil reviewer-response package on Colab Free T4
**Code baseline:** `55e1714` (the tree the reviewer-response work started from)
**Repo with fixes + full evidence:** https://github.com/swaroopms658/loralink-reviewer

---

## Bottom line

Four defects in LoraLink's model reconstruction meant the framework **could not
learn**. The worst of them disabled attention outright in half the GPT-Neo
layers. All four are fixed and verified against a reference forward pass, but
they are the code path that produced the paper's convergence figures and ΔPPL
ablation.

The **systems** contribution (speedup, bandwidth saved, partitioning) most
likely survives. The **quality and convergence** evidence does not.

---

## Check this first — it decides how much is affected

I verified these defects in the code at `55e1714`, and the paper states it used
this framework. I have **not** verified that the runs which produced
Fig. `loss_e2e`, Fig. `grid_convergence` and the ΔPPL table came from this exact
code state.

**If those figures were produced from a different branch or an earlier version,
the blast radius changes.** That is the cheapest possible next step and it
determines everything below.

---

## What was wrong

Each defect was only exposed by fixing the one before it, so the numbering is a
discovery order, not a priority ranking.

### 1. The forward pass skipped the final normalization layer

`pipeline_engine` composed the model as `wte → blocks → lm_head`. The reference
implementation is:

```
inputs_embeds   = wte(input_ids)
position_embeds = wpe(position_ids)        <- LoraLink: missing
hidden_states   = inputs_embeds + position_embeds
... transformer blocks ...
hidden_states   = ln_f(hidden_states)      <- LoraLink: missing
logits          = lm_head(hidden_states)
```

Without the final norm the residual stream reaches the unembedding
unnormalized, so logits are inflated by roughly two orders of magnitude.
Cross-entropy sat in the **thousands** instead of ~4–6.

Missing `wpe` additionally left GPT-Neo with **no positional signal at all** — it
uses learned absolute position embeddings, so the RoPE path never compensated.

Affects every architecture: GPT-Neo and Phi (`ln_f` / `final_layernorm`),
LLaMA/Mistral/Qwen2 (`model.norm`).

### 2. The loss was computed over padding

`data_loader` pads every sequence to 256 tokens. The pipeline built
`labels = input_ids.clone()` and called `cross_entropy` with no `ignore_index`,
so the loss averaged over ~256 positions that were mostly `<pad>`. The model
learned the trivial "emit padding" rule: 60 LoRA steps took the loss from 10.29
to **0.22**, far below what a 125M model reaches on real text.

This affected the training loss curve only. `eval_quality` scores one unpadded
text at a time, so reported perplexity was never contaminated by padding.

### 3. Attention was disabled in half the layers — the significant one

Blocks are built on the meta device, materialized with `to_empty()`, then filled
by `load_state_dict(..., strict=False, assign=True)`. That restores
*parameters*. It does **not** restore buffers registered `persistent=False`,
because those never appear in a checkpoint — so they keep whatever uninitialized
memory `to_empty()` left behind.

GPT-Neo's causal mask, `attn.attention.bias`, is exactly such a buffer, and it is
applied unconditionally:

```python
attn_weights = torch.where(causal_mask, attn_weights, mask_value)
```

Measured after materialization: **0 of 1024 cells unmasked** — every attention
weight replaced by the mask value.

The checkpoint happens to contain `bias` for the *global*-attention layers
`{0,2,4,6,8,10}`, so those were restored by accident. The six odd-numbered
*local*-attention layers had no checkpoint entry and stayed corrupt.
**Six of twelve layers ran with a garbage causal mask.**

With attention contributing nothing, the model predicted from token statistics
alone, which pins cross-entropy near the unigram entropy of English (~7.5 nats).
That is exactly what we measured, and it is why the compression ablation showed
nothing — attention was dead in every arm:

```
1 worker,  lossless    mean loss 7.604
2 workers, lossless    mean loss 7.321
2 workers, compressed  mean loss 7.533
```

### 4. On Phi, LoRA never adapted attention

`pipeline_engine` requested target modules `["Wqkv", "out_proj", "fc1", "fc2"]` —
the names used by the old *remote-code* `microsoft/phi-1_5`. The transformers
implementation names them `q_proj / k_proj / v_proj / dense` plus `fc1 / fc2`.

Only the two MLP projections matched, and unmatched names were skipped silently.
Phi runs adapted **2 modules per layer instead of 6**, with attention untouched.

GPT-Neo and LLaMA were checked and match their blocks correctly, so this one is
specific to the Phi experiments.

### 5. Seeds varied almost nothing (experimental design, not a bug)

The loader used `shuffle=False`, so all five seeds trained on the same samples in
the same order. The only seed-dependent quantity was LoRA's `A` initialization,
and since `B` starts at zero and 60 steps at lr 1e-4 barely move it, the runs
were near-identical: cross-seed std **0.0016** on a loss of 4.77.

That reports determinism, not the run-to-run robustness reviewers R3.4 / R3-Q7
asked for. Training now shuffles under a generator seeded from `--seed`.
Evaluation order is unchanged and batch size stays 1, per the paper's
hyperparameter table.

---

## Evidence

Each fix moved the loss, and the final state was checked against
`AutoModelForCausalLM` on identical batches — the check that would have caught
all of this at once.

| state | mean loss | reading |
|---|---|---|
| original code | **5322** | logits unnormalized; no learning |
| + final norm & `wpe` | 5.12 | right scale, but scoring padding (10.29 → 0.22) |
| + padding masked | 7.4, flat | attention dead; unigram entropy |
| + causal mask restored | **4.33** | matches reference |
| **HF reference (frozen)** | **4.510** | ground truth, same 30 batches |

The fixed pipeline coming in ~0.18 *below* the frozen reference is the LoRA
adapters actually learning over the measured batches.

Two results follow, and both are meaningful only now that attention works:

- **the pipeline hop is free** — 1 worker 4.331 vs 2 workers 4.329 (−0.002)
- **lossy compression costs +0.038 nats (+0.9 %)** — 4.329 → 4.367

Statistical validation after the shuffle fix, 5 seeds on WikiText-2:
mean **4.7331**, std **0.0195**, 95 % Student-t CI **[4.7089, 4.7574]**, n = 5.

---

## What this means for the paper

### Likely survives — but re-measure to confirm

These measure wall time and bytes moved, which do not depend on whether the
causal mask was correct:

- 2.4× training-time acceleration
- 93.6 % step-latency reduction
- 325.97–646.95 MB prevented per task; 2.8×–4.8× volume reduction
- Smart Partitioning behaviour and pipeline scaling

One caveat: sparsification is **magnitude-based**, so activations with a
different distribution compress differently. The mechanism holds; the exact
ratios should be re-measured.

### Invalid — produced with attention disabled

- Fig. `loss_e2e` and Fig. `grid_convergence` (all convergence trajectories)
- The ΔPPL ablation: 0.7625 % degradation on Dolly-15k; −2.09 % vs −2.02 % on E2E
- §542: *"rigorously validate that LoraLink's multi-hop pipeline parallelism does
  not degrade model convergence"*
- The LoRA Frobenius-norm analysis (A-norm 1.64–1.67), since Phi's attention
  adapters never existed

Worth noting: the reported ΔPPL values are *suspiciously small*, which is exactly
what adapters that barely learned would produce. That is corroborating evidence,
not a coincidence.

---

## Recommended next steps

1. **Confirm which code produced the published figures.** Everything above scales
   with the answer.
2. **Tell the co-authors before the rebuttal is drafted.** This gets worse the
   longer it waits, and far worse if a reviewer finds it.
3. **Re-measure the systems claims** to confirm they hold. The harness already
   exists.
4. **Regenerate the convergence and ΔPPL figures.** Larger job than the Colab
   package — the paper used 2.7 B models on 4 nodes.
5. **Consider disclosing it in the response.** *"We found and corrected a defect
   in our forward pass; here are the corrected numbers with the statistical
   validation you asked for"* reads as rigor. The same fact discovered by a
   reviewer reads as carelessness — and concerns 1 and 3 already require re-runs,
   so this is the natural moment.

---

## Where the raw evidence lives

All in https://github.com/swaroopms658/loralink-reviewer

| file | contents |
|---|---|
| `loralink_reviewer_response/patch/README.md` | per-defect write-up, with the reference-implementation comparisons |
| `loralink_reviewer_response/VERIFICATION.md` | the full defect ledger, including operational issues not covered here |
| `loralink_reviewer_response/patch/*.patch` | every source diff versus `55e1714` |
| `loralink_reviewer_response/tests/test_final_norm.py` | regression guard, defect 1 |
| `loralink_reviewer_response/tests/test_loss_masking.py` | regression guard, defect 2 |
| `loralink_reviewer_response/tests/test_nonpersistent_buffers.py` | regression guard, defect 3 |
| `loralink_reviewer_response/tests/test_target_modules.py` | regression guard, defect 4 |
| `loralink_reviewer_response/tests/test_seeded_shuffle.py` | regression guard, defect 5 |

Offline test suite: **131 passing**, up from 76 before this work.

### The common root, for whoever maintains this next

All four correctness defects come from the same design choice: the pipeline
reconstructs the model by hand — meta-device shells, `to_empty()`, and a partial
`state_dict` load — instead of going through HuggingFace's loader. That approach
is **silent by construction**. Anything the checkpoint does not contain — a
module the code forgot to create, a non-persistent buffer, a renamed submodule —
ends up absent or uninitialized with no error raised. The model still runs and
still produces a loss. It is simply the wrong loss.

Any future change to the reconstruction path should be validated against a
reference `AutoModelForCausalLM` forward on identical inputs.
