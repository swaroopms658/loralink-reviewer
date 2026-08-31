"""An adapter that loads as a no-op must fail, not report base-model numbers.

NB02 evaluated three separately trained arms and returned perplexity 42.4007 and
BLEU 4.291 for all three, to full precision. peft had built 144 lora_B tensors
from `adapter_config.json` and left every one of them at its zero init: the saved
weights never reached the model, so all three arms were the bare base model. The
run reported it as a compression ablation showing no quality cost.
"""
import types

import pytest
import torch

from loralink_reviewer_response import eval_quality


def _model_with(bs):
    """A stand-in exposing just the named_parameters() surface under test."""
    params = []
    for i, value in enumerate(bs):
        params.append((f"base_model.model.model.layers.{i}.mlp.fc1.lora_A.default.weight",
                       torch.full((2, 2), 0.02)))
        params.append((f"base_model.model.model.layers.{i}.mlp.fc1.lora_B.default.weight",
                       torch.full((2, 2), value)))
    return types.SimpleNamespace(named_parameters=lambda: iter(params))


def test_all_zero_lora_B_is_rejected():
    with pytest.raises(RuntimeError, match="no-op"):
        eval_quality._assert_adapter_is_live(_model_with([0.0, 0.0, 0.0]), "adapters/x")


def test_no_lora_parameters_at_all_is_rejected():
    with pytest.raises(RuntimeError, match="matched nothing"):
        eval_quality._assert_adapter_is_live(_model_with([]), "adapters/x")


def test_trained_adapter_passes():
    eval_quality._assert_adapter_is_live(_model_with([5.5e-3, 0.0, 2.1e-3]), "adapters/x")


def test_error_names_the_adapter_directory():
    with pytest.raises(RuntimeError, match="adapters/e2e_s0_ON"):
        eval_quality._assert_adapter_is_live(_model_with([0.0]), "adapters/e2e_s0_ON")


def test_save_warns_when_every_b_is_zero(capsys, tmp_path, monkeypatch):
    """The writing side should say so too, not only the reader."""
    import main

    weights = {
        "base_model.model.model.layers.0.mlp.fc1.lora_A.default.weight": torch.full((2, 2), 0.02),
        "base_model.model.model.layers.0.mlp.fc1.lora_B.default.weight": torch.zeros(2, 2),
    }
    main.save_lora_adapters(weights, lora_rank=8, output_path=str(tmp_path),
                            base_model_path="m", target_modules=["fc1"])
    assert "every lora_B is zero" in capsys.readouterr().out


def test_save_writes_safetensors_for_peft(tmp_path):
    """peft reads adapter_model.safetensors first; the .bin was being ignored."""
    import main

    weights = {
        "base_model.model.model.layers.0.mlp.fc1.lora_A.default.weight": torch.full((2, 2), 0.02),
        "base_model.model.model.layers.0.mlp.fc1.lora_B.default.weight": torch.full((2, 2), 5e-3),
    }
    main.save_lora_adapters(weights, lora_rank=8, output_path=str(tmp_path),
                            base_model_path="m", target_modules=["fc1"])
    assert (tmp_path / "adapter_model.safetensors").exists()
    assert (tmp_path / "adapter_config.json").exists()

    from safetensors.torch import load_file
    back = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert set(back) == set(weights)
    for k, v in weights.items():
        assert torch.equal(back[k], v)
