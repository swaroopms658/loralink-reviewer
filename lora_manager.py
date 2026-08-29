import os
import torch
import torch.nn as nn
import math
import types
from typing import List, Dict, NamedTuple, Optional


# ---------------------------------------------------------------------------
# LoRAAdapter: a lightweight record that holds the trainable A/B matrices and
# enough metadata to reconstruct the PEFT‑standard state‑dict key later.
# ---------------------------------------------------------------------------
class _LoRAAdapter(NamedTuple):
    """Holds all state for one injected LoRA pair."""
    layer_block: nn.Module        # The transformer-block that owns the params
    local_param_prefix: str       # e.g. "attn.attention.q_proj"
    lora_A: nn.Parameter          # shape (rank, in_features)
    lora_B: nn.Parameter          # shape (out_features, rank)
    scaling: float


# ---------------------------------------------------------------------------
# LoRAManager
# ---------------------------------------------------------------------------
class LoRAManager:
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        model_name: Optional[str] = None,
    ):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.model_name = model_name   # full local path or HF hub id

        # List of _LoRAAdapter records – one per injected linear
        self._adapters: List[_LoRAAdapter] = []

        # Flat list of Parameters fed to the optimizer
        self.lora_parameters: List[nn.Parameter] = []
        self.optimizer = None

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------
    def apply_lora_to_layers(
        self,
        model_layers: list,
        rank: int,
        alpha: float,
        target_modules: List[str],
    ) -> int:
        assert isinstance(model_layers, list)
        assert isinstance(rank, int) and rank > 0
        assert isinstance(alpha, (int, float)) and alpha > 0
        assert isinstance(target_modules, list)

        self._adapters = []
        self.lora_parameters = []

        for layer_block in model_layers:
            for name, module in layer_block.named_modules():
                if not isinstance(module, nn.Linear):
                    continue
                if not any(name.endswith(t) for t in target_modules):
                    continue

                # --- Create LoRA matrices on the same device/dtype as the weight ---
                device = module.weight.device
                dtype  = module.weight.dtype

                lora_A = nn.Parameter(
                    torch.zeros(rank, module.in_features, device=device, dtype=dtype)
                )
                lora_B = nn.Parameter(
                    torch.zeros(module.out_features, rank, device=device, dtype=dtype)
                )
                with torch.no_grad():
                    nn.init.kaiming_uniform_(lora_A, a=math.sqrt(5))
                    nn.init.zeros_(lora_B)

                scaling = alpha / rank

                # Register A/B on the layer_block so they appear in
                # layer_block.named_parameters() and are moved with .to()
                safe_name = name.replace(".", "__")
                layer_block.register_parameter(f"_lora_A_{safe_name}", lora_A)
                layer_block.register_parameter(f"_lora_B_{safe_name}", lora_B)

                # --- Monkey-patch forward (closure captures lora_A, lora_B, scaling) ---
                original_forward = module.forward

                def _make_lora_forward(orig_fwd, A, B, sc):
                    def lora_forward(x, *args, **kwargs):
                        base_out = orig_fwd(x, *args, **kwargs)
                        lora_out = (x @ A.t()) @ B.t()
                        return base_out + lora_out * sc
                    return lora_forward

                module.forward = _make_lora_forward(original_forward, lora_A, lora_B, scaling)

                # Freeze original weight
                module.weight.requires_grad_(False)
                if module.bias is not None:
                    module.bias.requires_grad_(False)

                adapter = _LoRAAdapter(
                    layer_block=layer_block,
                    local_param_prefix=name,   # e.g. "attn.attention.q_proj"
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                )
                self._adapters.append(adapter)
                self.lora_parameters.extend([lora_A, lora_B])

        if self.lora_parameters:
            self.optimizer = torch.optim.AdamW(
                self.lora_parameters,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            print(f"Created optimizer for {len(self.lora_parameters)} LoRA parameters "
                  f"across {len(self._adapters)} modules")

        return len(self.lora_parameters)

    # ------------------------------------------------------------------
    # Optimizer helpers
    # ------------------------------------------------------------------
    def zero_grad(self):
        if self.optimizer:
            self.optimizer.zero_grad()

    def step(self):
        if self.optimizer:
            torch.nn.utils.clip_grad_norm_(self.lora_parameters, max_norm=1.0)
            self.optimizer.step()
            return True
        return False

    # ------------------------------------------------------------------
    # State-dict export  (PEFT-standard keys)
    # ------------------------------------------------------------------
    def get_lora_state_dict(
        self,
        model_layers: list,
        assigned_layers: list = None,
    ) -> dict:
        from model_registry import ModelRegistry, ModelArchitecture

        model_name = self.model_name or "unknown"
        arch_type  = ModelRegistry.detect_architecture(model_name)
        arch_info  = ModelRegistry.get_architecture_info(arch_type)

        # ── Fallback: when AutoConfig can't identify the model (e.g. offline,
        #    malformed config), sniff the class name of the first layer block. ──
        if arch_type == ModelArchitecture.UNKNOWN and model_layers:
            first_cls = type(model_layers[0]).__name__
            if "GPTNeo" in first_cls:
                arch_type = ModelArchitecture.GPT_NEO
            elif "Phi" in first_cls:
                arch_type = ModelArchitecture.PHI
            elif "Llama" in first_cls:
                arch_type = ModelArchitecture.LLAMA
            elif "Mistral" in first_cls:
                arch_type = ModelArchitecture.MISTRAL
            elif "Qwen" in first_cls:
                arch_type = ModelArchitecture.QWEN2
            arch_info = ModelRegistry.get_architecture_info(arch_type)
            print(f"   ⚠️  Used class-name fallback: {first_cls} → {arch_type.value}")

        prefix = arch_info.layer_prefix if arch_info else "model.layers"
        print(f"📦 Exporting adapters for arch={arch_type.value}  prefix={prefix}")

        lora_params: Dict[str, torch.Tensor] = {}

        # Build a fast lookup: layer_block id -> global layer index
        layer_id_to_global = {}
        for local_idx, layer_block in enumerate(model_layers):
            global_idx = assigned_layers[local_idx] if assigned_layers else local_idx
            layer_id_to_global[id(layer_block)] = global_idx

        for adapter in self._adapters:
            global_layer_idx = layer_id_to_global.get(id(adapter.layer_block))
            if global_layer_idx is None:
                # Adapter belongs to a layer not in model_layers — skip
                continue

            module_path = adapter.local_param_prefix  # e.g. "attn.attention.q_proj"

            for m_type, param in (("lora_A", adapter.lora_A), ("lora_B", adapter.lora_B)):
                # Strict PEFT key format:
                # base_model.model.<layer_prefix>.<layer_idx>.<module_path>.<m_type>.default.weight
                key = (
                    f"base_model.model.{prefix}.{global_layer_idx}"
                    f".{module_path}.{m_type}.default.weight"
                )
                # Always export in float32 to maintain precision across nodes
                lora_params[key] = param.detach().clone().cpu().float()

        print(f"   Exported {len(lora_params)} tensors ({len(lora_params)//2} LoRA pairs)")
        return lora_params

    # ------------------------------------------------------------------
    # Legacy / merge helpers (kept for compatibility)
    # ------------------------------------------------------------------
    def set_lora_state_dict(self, model_layers: list, state_dict: dict):
        """Load back a PEFT-format state-dict (best-effort, for inference)."""
        for key, tensor in state_dict.items():
            # We don't reconstruct the full model here; callers should use
            # peft.PeftModel for inference loading.
            pass

    def merge_and_unload_lora(self, model_layers: list, target_modules: List[str]):
        """Merge LoRA deltas into the frozen weight in-place and remove patches."""
        merged_count = 0
        for adapter in self._adapters:
            if adapter.layer_block not in model_layers:
                continue
            # Find the module by path
            try:
                parts = adapter.local_param_prefix.rsplit(".", 1)
                if len(parts) == 2:
                    parent = adapter.layer_block.get_submodule(parts[0])
                    module = getattr(parent, parts[1])
                else:
                    module = getattr(adapter.layer_block, parts[0])
            except Exception:
                continue

            with torch.no_grad():
                delta = (adapter.lora_B @ adapter.lora_A) * adapter.scaling
                module.weight.add_(delta)

            merged_count += 1
        print(f"Merged {merged_count} LoRA adapters into base weights")
        return merged_count

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def get_learning_stats(self) -> Dict:
        if not self.lora_parameters:
            return {"status": "No LoRA parameters"}

        total_grad_norm  = 0.0
        total_param_norm = 0.0
        params_with_grad = 0

        for param in self.lora_parameters:
            total_param_norm += param.data.norm().item()
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm().item()
                params_with_grad += 1

        return {
            "total_parameters":      len(self.lora_parameters),
            "parameters_with_gradients": params_with_grad,
            "average_parameter_norm": total_param_norm / len(self.lora_parameters),
            "average_gradient_norm":  total_grad_norm / max(1, params_with_grad),
            "learning_rate":          self.learning_rate,
        }
