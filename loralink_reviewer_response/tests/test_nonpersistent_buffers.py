"""Buffers absent from the checkpoint must survive meta -> to_empty() loading.

`_load_model_parts` builds each block on the meta device, calls `to_empty()`, then
`load_state_dict(..., strict=False, assign=True)`. That restores parameters, but
NOT buffers registered with `persistent=False` -- those never appear in a
safetensors checkpoint, so `to_empty()` leaves them pointing at uninitialized
memory and nothing puts them back.

For GPT-Neo the casualty is `attn.attention.bias`, the lower-triangular causal
mask, applied unconditionally as
`torch.where(causal_mask, attn_weights, mask_value)`. Observed on Colab it
materialized all-False, masking every attention weight: attention contributed
nothing and the model collapsed to token statistics, pinning cross-entropy near
the unigram entropy of English (~7.5 nats) regardless of compression setting.
"""
import torch
from transformers import AutoConfig
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock

from pipeline_engine import restore_nonpersistent_buffers


def _cfg():
    cfg = AutoConfig.for_model(
        "gpt_neo", hidden_size=64, num_layers=2, num_heads=4,
        max_position_embeddings=32, attention_types=[[["global", "local"], 1]])
    cfg._attn_implementation = "eager"
    return cfg


def _meta_shell(cfg, layer_id=0):
    with torch.device("meta"):
        shell = GPTNeoBlock(cfg, layer_id=layer_id)
    shell.to_empty(device="cpu")
    return shell


def test_gpt_neo_causal_mask_is_non_persistent():
    """Guards the premise: the mask really is absent from the state dict."""
    ref = GPTNeoBlock(_cfg(), layer_id=0)
    assert "attn.attention.bias" not in ref.state_dict()
    assert "attn.attention.bias" in dict(ref.named_buffers())


def test_to_empty_corrupts_the_causal_mask():
    """Guards the premise: without a restore the mask is not the reference mask."""
    cfg = _cfg()
    ref = GPTNeoBlock(cfg, layer_id=0)
    shell = _meta_shell(cfg)
    assert not torch.equal(dict(shell.named_buffers())["attn.attention.bias"],
                           dict(ref.named_buffers())["attn.attention.bias"])


def test_restore_puts_the_causal_mask_back():
    cfg = _cfg()
    ref = GPTNeoBlock(cfg, layer_id=0)
    shell = _meta_shell(cfg)

    restore_nonpersistent_buffers(shell, ref, device=torch.device("cpu"))

    got = dict(shell.named_buffers())["attn.attention.bias"]
    want = dict(ref.named_buffers())["attn.attention.bias"]
    assert torch.equal(got, want)
    assert got.sum() > 0, "restored mask must leave some positions attendable"


def test_restore_is_a_copy_not_an_alias():
    """Shells must not share buffer storage with the reference or each other."""
    cfg = _cfg()
    ref = GPTNeoBlock(cfg, layer_id=0)
    shell = _meta_shell(cfg)
    restore_nonpersistent_buffers(shell, ref, device=torch.device("cpu"))
    assert (dict(shell.named_buffers())["attn.attention.bias"].data_ptr()
            != dict(ref.named_buffers())["attn.attention.bias"].data_ptr())


def test_restore_leaves_parameters_alone():
    cfg = _cfg()
    ref = GPTNeoBlock(cfg, layer_id=0)
    shell = _meta_shell(cfg)
    with torch.no_grad():
        for p in shell.parameters():
            p.fill_(0.25)
    restore_nonpersistent_buffers(shell, ref, device=torch.device("cpu"))
    for p in shell.parameters():
        assert torch.all(p == 0.25), "loaded weights must not be overwritten"


def test_local_and_global_attention_masks_differ():
    """GPT-Neo alternates global/local attention; with a real sliding window the
    two masks differ, so the reference must be built per layer, not shared."""
    cfg = _cfg()
    cfg.window_size = 4          # < max_position_embeddings, as in the real model
    g = dict(GPTNeoBlock(cfg, layer_id=0).named_buffers())["attn.attention.bias"]
    l = dict(GPTNeoBlock(cfg, layer_id=1).named_buffers())["attn.attention.bias"]
    assert not torch.equal(g, l), "local and global masks should differ"
    assert int(l.sum()) < int(g.sum()), "local attention should mask more"


def test_reference_block_always_has_real_buffers():
    """Whichever build path is taken, buffers must carry values (never meta)."""
    from model_registry import ModelArchitecture
    from pipeline_engine import build_reference_block

    cfg = _cfg()
    cfg.window_size = 4
    ref = build_reference_block(GPTNeoBlock, cfg, ModelArchitecture.GPT_NEO, 0)
    buffers = dict(ref.named_buffers())
    assert buffers, "reference block should expose buffers"
    for name, buf in buffers.items():
        assert buf.device.type != "meta", name
    want = dict(GPTNeoBlock(cfg, layer_id=0).named_buffers())["attn.attention.bias"]
    assert torch.equal(buffers["attn.attention.bias"], want)


def test_meta_reference_is_refused_rather_than_copied():
    """A meta-device reference must fail loudly, not corrupt the shell silently."""
    import pytest

    cfg = _cfg()
    shell = _meta_shell(cfg)
    with torch.device("meta"):
        meta_ref = GPTNeoBlock(cfg, layer_id=0)
    with pytest.raises(RuntimeError, match="meta device"):
        restore_nonpersistent_buffers(shell, meta_ref, device=torch.device("cpu"))


def test_restored_attention_actually_changes_the_output():
    cfg = _cfg()
    ref = GPTNeoBlock(cfg, layer_id=0)
    torch.manual_seed(0)
    x = torch.randn(1, 8, cfg.hidden_size)

    broken = _meta_shell(cfg)
    broken.load_state_dict(ref.state_dict(), strict=False, assign=True)
    fixed = _meta_shell(cfg)
    fixed.load_state_dict(ref.state_dict(), strict=False, assign=True)
    restore_nonpersistent_buffers(fixed, ref, device=torch.device("cpu"))

    with torch.no_grad():
        out_broken = broken(x)[0]
        out_fixed = fixed(x)[0]
        out_ref = ref(x)[0]

    assert torch.allclose(out_fixed, out_ref, atol=1e-5), "restored != reference"
    assert not torch.allclose(out_broken, out_ref, atol=1e-3), (
        "premise failed: corrupt mask should change the output")
