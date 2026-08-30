"""The pipeline must reproduce the reference forward pass.

Regression guard for a correctness bug: `pipeline_engine` composed
``wte -> blocks -> lm_head``, omitting the final normalization layer (``ln_f`` /
``model.norm`` / ``final_layernorm``) and, for GPT-Neo, the learned absolute
position embedding (``wpe``). Without the final norm the residual stream reaches
the unembedding unnormalized, logits blow up by ~2 orders of magnitude and
cross-entropy lands in the thousands instead of ~4-6, so nothing converges.
"""
import types

import pytest
import torch

from model_registry import ARCHITECTURE_PATTERNS, ModelArchitecture, ModelRegistry

_ROPE = [ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL,
         ModelArchitecture.QWEN2, ModelArchitecture.PHI]


def _cfg(hidden=64):
    return types.SimpleNamespace(
        hidden_size=hidden, layer_norm_epsilon=1e-5,
        layer_norm_eps=1e-5, rms_norm_eps=1e-6)


@pytest.mark.parametrize("arch", list(ARCHITECTURE_PATTERNS))
def test_every_architecture_declares_a_final_norm(arch):
    assert ARCHITECTURE_PATTERNS[arch].final_norm_prefix, arch


def test_final_norm_prefixes_match_reference_module_paths():
    got = {a: i.final_norm_prefix for a, i in ARCHITECTURE_PATTERNS.items()}
    assert got[ModelArchitecture.GPT_NEO] == "transformer.ln_f"
    assert got[ModelArchitecture.LLAMA] == "model.norm"
    assert got[ModelArchitecture.MISTRAL] == "model.norm"
    assert got[ModelArchitecture.QWEN2] == "model.norm"
    assert got[ModelArchitecture.PHI] == "model.final_layernorm"


def test_only_gpt_neo_has_learned_position_embeddings():
    # GPT-Neo adds wte + wpe; the RoPE architectures inject position at the block.
    assert (ARCHITECTURE_PATTERNS[ModelArchitecture.GPT_NEO].position_embedding_key
            == "transformer.wpe.weight")
    for arch in _ROPE:
        assert ARCHITECTURE_PATTERNS[arch].position_embedding_key is None, arch


@pytest.mark.parametrize("arch", list(ARCHITECTURE_PATTERNS))
def test_built_final_norm_actually_normalizes(arch):
    """A deep residual stream (std 50) must come out ~unit scale."""
    norm = ModelRegistry.build_final_norm(_cfg(), arch)
    with torch.no_grad():
        for name, p in norm.named_parameters():   # identity-init: weight 1, bias 0
            p.fill_(1.0 if name.endswith("weight") else 0.0)
        out = norm(torch.randn(2, 8, 64) * 50.0)
    assert out.std().item() < 2.0, f"{arch} final norm did not normalize: {out.std()}"


def test_pipeline_applies_final_norm_before_lm_head():
    import inspect

    import pipeline_engine

    for fn in (pipeline_engine.PipelineStage.forward_step_local,
               pipeline_engine.PipelineStage.forward_step_remote):
        src = inspect.getsource(fn)
        assert "self.final_norm" in src, fn.__name__
        head = src.index("self.lm_head(")
        assert "final_norm" in src[:head], (
            f"{fn.__name__} calls lm_head without normalizing first")
