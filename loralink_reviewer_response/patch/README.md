# patch/ — source changes versus the paper's original code

Every `<file>.patch` here is `git diff 55e1714 -- <file>` — the diff against the
commit the paper's results were produced from — and exists so a reviewer can see
exactly what changed. The patched tree ships directly: the Colab notebooks clone
this branch with the changes already merged, so nothing needs to be applied.
`apply_patch.py --repo <dir>` just runs `checksums.py --verify`, which compares
the live sources against `SHA256SUMS` and exits non-zero on any mismatch.
Regenerate the baseline with `python checksums.py --update`.

The changes fall into **two categories that must not be conflated**.

## 1. Additive instrumentation (5 files)

`main.py`, `device_manager.py`, `compression_engine.py`, `benchmarking.py`,
`data_loader.py`.

New CLI flags, seeding, bind-address selection, a lossless-only compression
toggle, extra partitioner strategies, a metrics CSV writer, an evaluation
hold-out split, and a benchmark fast-path. None of it touches the optimizer,
the LoRA math, the compression math, or the wire format. Default behaviour is
unchanged. This is the reviewer-response harness proper.

## 1b. Seeded shuffling of the training order

`data_loader.py`. Listed separately because it is an experimental-design change
rather than a bug fix or pure instrumentation.

The loader built its `DataLoader` with `shuffle=False`, so every seed trained on
the same samples in the same order. The only seed-dependent quantity was LoRA's
`A` initialization, and because `B` starts at zero and 60 steps at lr 1e-4 barely
move it, five seeds produced a cross-seed std of **0.0016** on a loss of 4.77 —
evidence that the code is deterministic, not that the results are robust, which
is what R3.4 / R3-Q7 actually ask for.

`get_data_loader` now takes `seed=` and shuffles the **training** split under a
`torch.Generator` seeded from `--seed`. Evaluation is never shuffled, so held-out
metrics stay comparable across runs, and batch size remains 1 as specified in the
paper's hyperparameter table. The paper reports single runs per configuration and
explicitly defers variance reporting, so no published number is contradicted.

Regression guard: `tests/test_seeded_shuffle.py`.

## 2. Forward-pass correctness fix (2 files)

`pipeline_engine.py`, `model_registry.py`.

**This is a behaviour change, not instrumentation, and it changes results.**

`PipelineStage` composed the forward pass as `wte → blocks → lm_head`, omitting
two pieces of the reference implementation:

| | Reference (`GPTNeoModel.forward`) | Before this fix |
|---|---|---|
| embedding | `wte(ids) + wpe(pos)` | `wte(ids)` — no `wpe` |
| final norm | `ln_f(hidden)` before the head | *absent* |

Consequences:

- **Missing final norm** (`ln_f` for GPT-Neo/Phi, `model.norm` for
  LLaMA/Mistral/Qwen2) left the residual stream unnormalized at the unembedding.
  Logits were inflated by roughly two orders of magnitude, so cross-entropy
  landed in the **thousands** instead of ~4–6. Measured on Colab T4 with
  pretrained `gpt-neo-125M`: loss oscillated 2 000–11 000 with no learning trend,
  and was **identical with compression ON and OFF** (mean 6 592 vs 6 443),
  which is what ruled out the compression engine as the cause.
- **Missing `wpe`** left GPT-Neo with *no positional signal at all* — it uses
  learned absolute position embeddings, and `needs_position_ids` is correctly
  `False` for it, so the RoPE path never compensated. RoPE architectures
  (LLaMA/Mistral/Qwen2/Phi) were affected only by the missing final norm.

### 2b. Padding excluded from the loss

Same file, same category. `data_loader` pads every sequence to `max_length=256`,
but the pipeline built `labels = input_ids.clone()` with no mask and called
`cross_entropy` without `ignore_index`. The loss therefore averaged over ~256
positions that are mostly PAD, and the model learned the trivial "emit padding"
rule: after the norm fix, 60 LoRA steps on wikitext drove the loss from 10.29 to
**0.22** — far below what a 125 M model reaches on real text, because most of
what it was scoring was padding.

Fixed with `build_masked_labels()` (padded positions → `IGNORE_INDEX = -100`,
applied at rank 0 so the already-masked labels are what travels to remote
stages) and `ignore_index=IGNORE_INDEX` at both `cross_entropy` call sites.
The sentinel round-trips losslessly through the `labels` compression path
(sparsity 0.0, no quantization) — guarded by
`tests/test_loss_masking.py::test_ignore_index_survives_the_wire`.

Note this affects the **training loss curve only**. `eval_quality._perplexity`
scores one unpadded text at a time, so the reported perplexity, BLEU and ROUGE-L
were never contaminated by padding.

### 2c. Non-persistent buffers destroyed by `to_empty()`

Same file, same category, and the one that actually dominated the loss.

Blocks are built on the meta device, materialized with `to_empty()`, then filled
by `load_state_dict(..., strict=False, assign=True)`. That restores *parameters*.
It does not restore buffers registered `persistent=False`, because those never
appear in a checkpoint — so they keep whatever `to_empty()` left in memory.

GPT-Neo's causal mask, `attn.attention.bias`, is exactly such a buffer, and it is
applied unconditionally inside `_attn`:

```python
attn_weights = torch.where(causal_mask, attn_weights, mask_value)
```

Measured after materialization: **0 of 1024 cells unmasked** — every attention
weight replaced by the mask value. Attention contributed nothing, leaving the
model to predict from token statistics alone, which pins cross-entropy near the
unigram entropy of English (~7.5 nats). That matches the measurement precisely,
and explains why it did not move with compression:

```
1 worker,  lossless   mean loss 7.604
2 workers, lossless   mean loss 7.321
2 workers, compressed mean loss 7.533
```

Attention was dead in all three arms, so the ablation was measuring nothing.

Fixed by `build_reference_block()` + `restore_nonpersistent_buffers()`: after
each layer loads, a reference block is constructed and its non-persistent buffers
copied in. The reference is built **per layer** — GPT-Neo alternates global and
local attention and the two masks differ — and via
`accelerate.init_empty_weights(include_buffers=False)` so its parameters stay on
meta and only buffers cost memory (a fully materialized reference would add
~216 MB per layer for Phi-1.5). Both the builder and the copier refuse a
meta-device buffer rather than copy it, so a change in accelerate's behaviour
fails loudly instead of silently restoring the corruption.

Regression guard: `tests/test_nonpersistent_buffers.py`.

### Implementation

The final-norm fix adds `final_norm_prefix`, `final_norm_class_name` and
`position_embedding_key` to `ArchitectureInfo`, a `ModelRegistry.build_final_norm`
factory, loading for both modules, and their application in
`forward_step_local` / `forward_step_remote`. Both are **frozen** base-model
parts (`requires_grad_(False)`) and are not LoRA targets — `apply_lora_to_layers`
only ever sees `self.layers` — so the optimizer and LoRA math are untouched.

Regression guards: `tests/test_final_norm.py`, `tests/test_loss_masking.py`,
`tests/test_nonpersistent_buffers.py`.

### Why three defects, not one

All three share a root: the pipeline reconstructs the model by hand — meta-device
shells, `to_empty()`, and a partial `state_dict` load — instead of going through
HuggingFace's loader. That approach is silent by construction. Anything the
checkpoint does not contain (a module the code forgot to create, a
non-persistent buffer) ends up either absent or uninitialized, with no error
raised. The model still runs and still produces a loss; it is simply the wrong
loss. A future change to the reconstruction path should be treated as
high-risk and validated against a reference `AutoModelForCausalLM` forward on
identical inputs, which is the check that would have caught all three at once.

### Bearing on the paper

The paper's convergence figures, ΔPPL ablation, and the claim that pipeline
parallelism "does not degrade model convergence" were produced by the code
*before* this fix. Any number in the paper that depends on the forward pass
should be regenerated. This is flagged rather than silently corrected: the
decision about the paper's existing results belongs to the authors.
