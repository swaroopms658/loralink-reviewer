"""LoRA target-module names must match the architecture actually being loaded.

`pipeline_engine` selected `["Wqkv", "out_proj", "fc1", "fc2"]` for Phi -- the
module names of the old *remote-code* `microsoft/phi-1_5`. The transformers
implementation used here names them `q_proj/k_proj/v_proj/dense` plus
`fc1/fc2`, so only the two MLP projections matched and **attention was never
adapted**: NB02 saved 48 LoRA pairs (2 per layer x 24) instead of 6 per layer.
`apply_lora_to_layers` skips non-matching names silently, so nothing complained.

GPT-Neo and LLaMA were checked and match correctly.
"""
import torch
from transformers import AutoConfig


def _linear_names(block):
    return {n.split(".")[-1] for n, m in block.named_modules()
            if isinstance(m, torch.nn.Linear)}


def _phi_block():
    from transformers.models.phi.modeling_phi import PhiDecoderLayer
    cfg = AutoConfig.for_model("phi", hidden_size=64, num_hidden_layers=2,
                               num_attention_heads=4, intermediate_size=128,
                               max_position_embeddings=64)
    cfg._attn_implementation = "eager"
    return PhiDecoderLayer(cfg, layer_idx=0)


def _neo_block():
    from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
    cfg = AutoConfig.for_model("gpt_neo", hidden_size=64, num_layers=2, num_heads=4,
                               max_position_embeddings=32,
                               attention_types=[[["global", "local"], 1]])
    cfg._attn_implementation = "eager"
    return GPTNeoBlock(cfg, layer_id=0)


def test_phi_targets_cover_attention_and_mlp():
    from pipeline_engine import target_modules_for
    from model_registry import ModelArchitecture

    targets = set(target_modules_for(ModelArchitecture.PHI))
    actual = _linear_names(_phi_block())
    unadapted = actual - targets
    assert not unadapted, f"Phi leaves these unadapted: {sorted(unadapted)}"
    assert {"q_proj", "k_proj", "v_proj", "dense"} <= targets, "attention must be adapted"


def test_gpt_neo_targets_still_cover_everything():
    from pipeline_engine import target_modules_for
    from model_registry import ModelArchitecture

    targets = set(target_modules_for(ModelArchitecture.GPT_NEO))
    assert not (_linear_names(_neo_block()) - targets)


def test_unmatched_targets_are_reported(capsys):
    """A name that matches nothing must be loud, not silently skipped."""
    from lora_manager import LoRAManager

    block = _neo_block()
    mgr = LoRAManager(model_name="x")
    mgr.apply_lora_to_layers(model_layers=[block], rank=4, alpha=8,
                             target_modules=["q_proj", "not_a_real_module"])
    out = capsys.readouterr().out
    assert "not_a_real_module" in out, "unmatched target module must be reported"


def test_matching_targets_do_not_warn(capsys):
    from lora_manager import LoRAManager

    block = _neo_block()
    mgr = LoRAManager(model_name="x")
    mgr.apply_lora_to_layers(model_layers=[block], rank=4, alpha=8,
                             target_modules=["q_proj", "v_proj"])
    assert "matched no modules" not in capsys.readouterr().out


def test_phi_adapter_count_would_be_six_per_layer():
    """Regression on the observed symptom: 2 pairs/layer instead of 6."""
    from lora_manager import LoRAManager
    from model_registry import ModelArchitecture
    from pipeline_engine import target_modules_for

    block = _phi_block()
    mgr = LoRAManager(model_name="x")
    mgr.apply_lora_to_layers(
        model_layers=[block], rank=4, alpha=8,
        target_modules=target_modules_for(ModelArchitecture.PHI))
    # one A and one B per adapted Linear
    assert len(mgr.lora_parameters) == 2 * 6, len(mgr.lora_parameters)
