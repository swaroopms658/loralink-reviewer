"""Padding must not contribute to the training loss.

`data_loader` pads every sequence to max_length=256, and the pipeline built
`labels = input_ids.clone()` with no mask, so cross-entropy averaged over ~256
positions that are mostly PAD. The model then trivially learns "emit padding"
and the loss collapses -- measured on Colab, 60 LoRA steps took wikitext loss
from 10.29 to 0.22, which is far below what a 125M model reaches on real text.
That curve is an artefact, not convergence.
"""
import inspect

import torch

from pipeline_engine import IGNORE_INDEX, build_masked_labels


def test_pad_positions_become_ignore_index():
    input_ids = torch.tensor([[5, 6, 7, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
    labels = build_masked_labels(input_ids, attention_mask)
    assert labels.tolist() == [[5, 6, 7, IGNORE_INDEX, IGNORE_INDEX]]


def test_real_tokens_are_untouched():
    input_ids = torch.tensor([[11, 12, 13]])
    attention_mask = torch.ones_like(input_ids)
    assert build_masked_labels(input_ids, attention_mask).tolist() == [[11, 12, 13]]


def test_missing_mask_is_a_no_op():
    input_ids = torch.tensor([[1, 2, 3]])
    assert build_masked_labels(input_ids, None).tolist() == [[1, 2, 3]]


def test_input_ids_not_mutated():
    input_ids = torch.tensor([[5, 6, 0]])
    build_masked_labels(input_ids, torch.tensor([[1, 1, 0]]))
    assert input_ids.tolist() == [[5, 6, 0]], "must not write through to input_ids"


def test_masked_loss_ignores_padding():
    """A prediction that is wrong only on padded positions must cost nothing."""
    import torch.nn.functional as F

    vocab = 7
    logits = torch.zeros(1, 3, vocab)
    logits[0, 0, 5] = 50.0   # confident + correct on the one real target
    logits[0, 1, 0] = 50.0   # confident + wrong, but this position is padding
    labels = build_masked_labels(torch.tensor([[9, 5, 0]]),
                                 torch.tensor([[1, 1, 0]]))
    shift_logits = logits[:, :-1, :].reshape(-1, vocab)
    shift_labels = labels[:, 1:].reshape(-1)
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
    assert loss.item() < 0.01, loss.item()


def test_ignore_index_survives_the_wire():
    """Labels are compressed and shipped to remote stages; -100 must round-trip."""
    from compression_engine import OptimizedCompressionEngine

    engine = OptimizedCompressionEngine()
    labels = torch.tensor([[5, 6, 7, IGNORE_INDEX], [1, 2, IGNORE_INDEX, IGNORE_INDEX]],
                          dtype=torch.long)
    back = engine.decompress_tensor(engine.compress_tensor(labels, tensor_type="labels"))
    assert torch.equal(back, labels), back


def test_both_forward_paths_pass_ignore_index():
    import pipeline_engine

    for fn in (pipeline_engine.PipelineStage.forward_step_local,
               pipeline_engine.PipelineStage.forward_step_remote):
        src = inspect.getsource(fn)
        assert "cross_entropy" in src, fn.__name__
        assert "ignore_index=IGNORE_INDEX" in src, (
            f"{fn.__name__} computes cross-entropy without masking padding")
