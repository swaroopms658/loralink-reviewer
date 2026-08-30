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

### Implementation

The final-norm fix adds `final_norm_prefix`, `final_norm_class_name` and
`position_embedding_key` to `ArchitectureInfo`, a `ModelRegistry.build_final_norm`
factory, loading for both modules, and their application in
`forward_step_local` / `forward_step_remote`. Both are **frozen** base-model
parts (`requires_grad_(False)`) and are not LoRA targets — `apply_lora_to_layers`
only ever sees `self.layers` — so the optimizer and LoRA math are untouched.

Regression guards: `tests/test_final_norm.py`, `tests/test_loss_masking.py`.

### Bearing on the paper

The paper's convergence figures, ΔPPL ablation, and the claim that pipeline
parallelism "does not degrade model convergence" were produced by the code
*before* this fix. Any number in the paper that depends on the forward pass
should be regenerated. This is flagged rather than silently corrected: the
decision about the paper's existing results belongs to the authors.
