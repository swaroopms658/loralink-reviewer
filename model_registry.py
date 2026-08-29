"""
Model Registry for LoraLink

Provides architecture detection and model-specific configuration for different
transformer models (GPT-Neo, LLaMA, Mistral, Qwen, Phi, etc.)
"""

import enum
from dataclasses import dataclass
from typing import Optional, Dict, Any, Type
import transformers
from transformers import PretrainedConfig


class ModelArchitecture(enum.Enum):
    """Supported model architectures"""
    GPT_NEO = "gpt_neo"
    LLAMA = "llama"  # Covers LLaMA 1/2/3 and Open LLaMA
    MISTRAL = "mistral"
    QWEN2 = "qwen2"  # Covers Qwen2 and Qwen2.5
    PHI = "phi"  # Covers Phi-2
    UNKNOWN = "unknown"


@dataclass
class ArchitectureInfo:
    """Architecture-specific information"""
    architecture: ModelArchitecture
    layer_prefix: str  # e.g., "transformer.h" or "model.layers"
    embedding_key: str  # e.g., "transformer.wte.weight"
    lm_head_key: str  # e.g., "lm_head.weight"
    num_layers_attr: str  # config attribute name for layer count
    hidden_size_attr: str  # config attribute name for hidden dimension
    block_class_name: str  # Name of the transformer block class
    uses_tied_embeddings: bool  # Whether LM head shares weights with embeddings


# Architecture patterns
ARCHITECTURE_PATTERNS = {
    ModelArchitecture.GPT_NEO: ArchitectureInfo(
        architecture=ModelArchitecture.GPT_NEO,
        layer_prefix="transformer.h",
        embedding_key="transformer.wte.weight",
        lm_head_key="transformer.wte.weight",  # Tied to embeddings
        num_layers_attr="num_layers",
        hidden_size_attr="hidden_size",
        block_class_name="GPTNeoBlock",
        uses_tied_embeddings=True
    ),
    ModelArchitecture.LLAMA: ArchitectureInfo(
        architecture=ModelArchitecture.LLAMA,
        layer_prefix="model.layers",
        embedding_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        num_layers_attr="num_hidden_layers",
        hidden_size_attr="hidden_size",
        block_class_name="LlamaDecoderLayer",
        uses_tied_embeddings=False
    ),
    ModelArchitecture.MISTRAL: ArchitectureInfo(
        architecture=ModelArchitecture.MISTRAL,
        layer_prefix="model.layers",
        embedding_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        num_layers_attr="num_hidden_layers",
        hidden_size_attr="hidden_size",
        block_class_name="MistralDecoderLayer",
        uses_tied_embeddings=False
    ),
    ModelArchitecture.QWEN2: ArchitectureInfo(
        architecture=ModelArchitecture.QWEN2,
        layer_prefix="model.layers",
        embedding_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        num_layers_attr="num_hidden_layers",
        hidden_size_attr="hidden_size",
        block_class_name="Qwen2DecoderLayer",
        uses_tied_embeddings=False
    ),
    ModelArchitecture.PHI: ArchitectureInfo(
        architecture=ModelArchitecture.PHI,
        layer_prefix="model.layers",
        embedding_key="model.embed_tokens.weight",
        lm_head_key="lm_head.weight",
        num_layers_attr="num_hidden_layers",
        hidden_size_attr="hidden_size",
        block_class_name="PhiDecoderLayer",
        uses_tied_embeddings=False
    ),
}


class ModelRegistry:
    """Registry for detecting and managing different model architectures"""
    
    @staticmethod
    def detect_architecture(model_path_or_name: str) -> ModelArchitecture:
        """
        Detect model architecture from HuggingFace config
        
        Args:
            model_path_or_name: Path to model directory or HF model name
            
        Returns:
            ModelArchitecture enum value
        """
        try:
            config = transformers.AutoConfig.from_pretrained(model_path_or_name)
            model_type = config.model_type.lower()
            
            # Map HuggingFace model_type to our architecture enum
            architecture_map = {
                "gpt_neo": ModelArchitecture.GPT_NEO,
                "llama": ModelArchitecture.LLAMA,
                "mistral": ModelArchitecture.MISTRAL,
                "qwen2": ModelArchitecture.QWEN2,
                "phi": ModelArchitecture.PHI,
            }
            
            return architecture_map.get(model_type, ModelArchitecture.UNKNOWN)
            
        except Exception as e:
            print(f"Error detecting architecture for {model_path_or_name}: {e}")
            return ModelArchitecture.UNKNOWN
    
    @staticmethod
    def get_architecture_info(architecture: ModelArchitecture) -> Optional[ArchitectureInfo]:
        """Get architecture-specific information"""
        return ARCHITECTURE_PATTERNS.get(architecture)
    
    @staticmethod
    def get_num_layers(config: PretrainedConfig, architecture: ModelArchitecture) -> int:
        """Extract number of layers from config based on architecture"""
        arch_info = ARCHITECTURE_PATTERNS.get(architecture)
        if arch_info is None:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return getattr(config, arch_info.num_layers_attr, 32)
    
    @staticmethod
    def get_hidden_size(config: PretrainedConfig, architecture: ModelArchitecture) -> int:
        """Extract hidden size from config based on architecture"""
        arch_info = ARCHITECTURE_PATTERNS.get(architecture)
        if arch_info is None:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        return getattr(config, arch_info.hidden_size_attr, 2560)
    
    @staticmethod
    def load_block_class(architecture: ModelArchitecture) -> Type:
        """
        Dynamically load the transformer block class for the architecture
        
        Returns:
            The transformer block class (e.g., GPTNeoBlock, LlamaDecoderLayer)
        """
        arch_info = ARCHITECTURE_PATTERNS.get(architecture)
        if arch_info is None:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Map architecture to module path and class name
        module_map = {
            ModelArchitecture.GPT_NEO: "transformers.models.gpt_neo.modeling_gpt_neo",
            ModelArchitecture.LLAMA: "transformers.models.llama.modeling_llama",
            ModelArchitecture.MISTRAL: "transformers.models.mistral.modeling_mistral",
            ModelArchitecture.QWEN2: "transformers.models.qwen2.modeling_qwen2",
            ModelArchitecture.PHI: "transformers.models.phi.modeling_phi",
        }
        
        module_path = module_map.get(architecture)
        if module_path is None:
            raise ValueError(f"No module mapping for architecture: {architecture}")
        
        # Dynamic import
        import importlib
        module = importlib.import_module(module_path)
        block_class = getattr(module, arch_info.block_class_name)
        
        return block_class
    
    @staticmethod
    def estimate_model_size(config: PretrainedConfig, architecture: ModelArchitecture) -> Dict[str, float]:
        """
        Estimate model memory requirements for training
        
        Returns:
            Dict with 'total_gb', 'per_layer_gb', 'num_layers'
        """
        arch_info = ARCHITECTURE_PATTERNS.get(architecture)
        if arch_info is None:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        num_layers = ModelRegistry.get_num_layers(config, architecture)
        hidden_size = ModelRegistry.get_hidden_size(config, architecture)
        vocab_size = getattr(config, 'vocab_size', 50257)
        intermediate_size = getattr(config, 'intermediate_size', 4 * hidden_size)
        
        # Architecture-aware parameter estimation per layer
        if architecture in [ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL, ModelArchitecture.QWEN2, ModelArchitecture.PHI]:
            # GQA attention: Q/O use full hidden, K/V use smaller kv_dim
            num_heads = getattr(config, 'num_attention_heads', 32)
            num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
            head_dim = hidden_size // num_heads
            kv_dim = num_kv_heads * head_dim
            
            # Gated MLP (gate_proj + up_proj + down_proj)
            params_per_layer = (
                hidden_size * hidden_size      # q_proj
                + hidden_size * kv_dim         # k_proj
                + hidden_size * kv_dim         # v_proj
                + hidden_size * hidden_size    # o_proj
                + hidden_size * intermediate_size * 3  # gate + up + down
            )
        else:
            # Standard transformer (GPT-Neo, etc.): ~12 * hidden^2
            params_per_layer = 12 * (hidden_size ** 2)
        
        total_params = (params_per_layer * num_layers) + (vocab_size * hidden_size)
        
        # Convert to GB (4 bytes per float32 parameter)
        model_size_gb = (total_params * 4) / (1024 ** 3)
        
        # Training overhead: activations stored for backward, autograd graph, etc.
        effective_size_gb = model_size_gb * 1.80
        per_layer_gb = effective_size_gb / num_layers
        
        return {
            'total_gb': effective_size_gb,
            'per_layer_gb': per_layer_gb,
            'num_layers': num_layers
        }


if __name__ == "__main__":
    # Test the registry
    registry = ModelRegistry()
    
    # Test architecture detection
    test_models = [
        "./models/EleutherAI/gpt-neo-2.7B",
        # Add more if you have them locally
    ]
    
    for model_path in test_models:
        print(f"\nTesting: {model_path}")
        try:
            arch = registry.detect_architecture(model_path)
            print(f"  Architecture: {arch}")
            
            if arch != ModelArchitecture.UNKNOWN:
                info = registry.get_architecture_info(arch)
                print(f"  Layer prefix: {info.layer_prefix}")
                print(f"  Embedding key: {info.embedding_key}")
                print(f"  Block class: {info.block_class_name}")
                
                config = transformers.AutoConfig.from_pretrained(model_path)
                size_info = registry.estimate_model_size(config, arch)
                print(f"  Estimated size: {size_info['total_gb']:.2f} GB")
                print(f"  Per layer: {size_info['per_layer_gb']:.4f} GB")
                
        except Exception as e:
            print(f"  Error: {e}")
