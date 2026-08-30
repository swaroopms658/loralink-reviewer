import torch
import torch.nn as nn
import torch.nn.functional as F
from lora_manager import LoRAManager
from device_manager import PipelineConfig
from compression_engine import OptimizedCompressionEngine
from network_protocol import NetworkManager, Message, MessageType
import transformers
import base64
import time
import os
from typing import Dict, Any, Tuple, Optional, Type
from safetensors import safe_open
import gc
import threading
from model_registry import ModelRegistry, ModelArchitecture, ArchitectureInfo

# Label value that cross_entropy skips. data_loader pads to max_length, so without
# this the loss averages over mostly-PAD positions and the model just learns to
# emit padding -- the loss collapses toward 0 without learning the language.
IGNORE_INDEX = -100


def build_masked_labels(input_ids: torch.Tensor,
                        attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Next-token labels with padded positions set to IGNORE_INDEX.

    Returns a fresh tensor; `input_ids` is never written through.
    """
    labels = input_ids.clone()
    if attention_mask is None:
        return labels
    return labels.masked_fill(attention_mask.to(labels.device) == 0, IGNORE_INDEX)


class PipelineStage:
    def __init__(self, config: PipelineConfig, network_manager: NetworkManager):
        # 1. SETUP DEVICE DYNAMICALLY (Supports CPU, GPU, or Mixed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Pipeline Engine initialized on: {self.device}")
        
        self.config = config
        self.network_manager = network_manager
        self.compression_engine = OptimizedCompressionEngine()  
            
        # model_name is stored as the full local path so ModelRegistry can resolve
        # it offline via AutoConfig.from_pretrained without hitting the HF hub.
        local_model_path = f"./models/{config.model_name}"
        self.lora_manager = LoRAManager(
            learning_rate=1e-4,
            weight_decay=0.01,
            model_name=local_model_path,
        )

        # 2. LOAD MODEL PARTS
        # final_norm / position_embedding are frozen base-model parts, not LoRA
        # targets: apply_lora_to_layers below only ever sees self.layers.
        (self.layers, self.embedding_layer, self.lm_head,
         self.final_norm, self.position_embedding) = self._load_model_parts()

        # Architecture-aware target modules — cover attention AND MLP projections.
        if self.architecture == ModelArchitecture.GPT_NEO:
            # GPT-Neo: attention q/k/v/out + MLP c_fc/c_proj
            target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"]
        elif self.architecture == ModelArchitecture.PHI:
            # Phi-2: fused QKV + attention out + MLP fc1/fc2
            target_modules = ["Wqkv", "out_proj", "fc1", "fc2"]
        else:
            # LLaMA / Mistral / Qwen2: attention q/k/v/o + gated MLP
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj"]

        # list(self.layers) gives individual block objects so adapter object-identity
        # matching in get_lora_state_dict works correctly (not the container).
        num_params = self.lora_manager.apply_lora_to_layers(
            model_layers=list(self.layers),
            rank=config.lora_rank,
            alpha=16.0,
            target_modules=target_modules,
        )
        self._target_modules = target_modules   # stored for adapter_config export
        
        self.forward_cache = {}
        self.cache_lock = threading.Lock()
        self.max_cache_size = 3  
        print(f"DEBUG PIPELINE: Assigned layers: {config.assigned_layers}")
        print(f"PipelineStage initialized: {len(config.assigned_layers)} layers, {num_params} LoRA parameters")

    def _cleanup_old_cache_entries(self):
        with self.cache_lock:
            if len(self.forward_cache) > self.max_cache_size:
                sorted_keys = sorted(self.forward_cache.keys())
                keys_to_remove = sorted_keys[:-self.max_cache_size//2]
                for key in keys_to_remove:
                    if key in self.forward_cache:
                        del self.forward_cache[key]
                gc.collect()

    def _load_model_parts(self) -> Tuple[nn.ModuleList, Optional[nn.Module], Optional[nn.Module],
                                         Optional[nn.Module], Optional[nn.Module]]:
        # Use model_name from config (passed from device_manager)
        model_path = f"./models/{self.config.model_name}"
        safetensors_path = f"{model_path}/model.safetensors"
        
        print(f"Loading model config from {model_path}")
        model_config = transformers.AutoConfig.from_pretrained(model_path)
        model_config._attn_implementation = "eager"
        
        # Store model_config for later use in RoPE calculations
        self.model_config = model_config
        
        # Detect architecture
        architecture = ModelRegistry.detect_architecture(model_path)
        arch_info = ModelRegistry.get_architecture_info(architecture)
        
        if arch_info is None:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        print(f"Architecture: {architecture.value}")
        
        # Store architecture info for later use  
        self.architecture = architecture
        self.needs_position_ids = architecture in [
            ModelArchitecture.LLAMA,
            ModelArchitecture.MISTRAL,
            ModelArchitecture.QWEN2,
            ModelArchitecture.PHI
        ]
        
        # FP32 mode: all tensors stay in native float32 precision
        
        print(f"Creating empty model shells for assigned layers: {self.config.assigned_layers}")
        
        # Dynamically load the appropriate block class
        block_class = ModelRegistry.load_block_class(architecture)
        
        # Get num_layers for validation
        num_layers = ModelRegistry.get_num_layers(model_config, architecture)
        
        # Create empty layer shells
        layers = nn.ModuleList()
        total_assigned = len(self.config.assigned_layers)
        for shell_idx, layer_idx in enumerate(self.config.assigned_layers):
            assert 0 <= layer_idx < num_layers, f"Layer {layer_idx} out of range [0, {num_layers})"
            
            # Create layer shell on meta device (zero memory allocation)
            with torch.device('meta'):
                if architecture == ModelArchitecture.GPT_NEO:
                    empty_layer = block_class(model_config, layer_id=layer_idx)
                else:
                    # LLaMA, Mistral, Qwen, Phi use layer_idx parameter
                    empty_layer = block_class(model_config, layer_idx=layer_idx)

            layers.append(empty_layer)
            
            if (shell_idx + 1) % 10 == 0 or (shell_idx + 1) == total_assigned:
                print(f"   Created {shell_idx + 1}/{total_assigned} layer shells (meta device)")
        
        # Create embedding layer (only for rank 0) on meta device
        embedding_layer = None
        if self.config.device_rank == 0:
            hidden_size = ModelRegistry.get_hidden_size(model_config, architecture)
            vocab_size = getattr(model_config, 'vocab_size', 50257)
            with torch.device('meta'):
                embedding_layer = nn.Embedding(vocab_size, hidden_size)

        
        # Create LM head (only for last device in pipeline) on meta device
        lm_head = None
        if self.config.successor_ip is None:
            hidden_size = ModelRegistry.get_hidden_size(model_config, architecture)
            vocab_size = getattr(model_config, 'vocab_size', 50257)
            with torch.device('meta'):
                lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Create the final normalization layer (last device only). The reference
        # forward is `hidden = ln_f(hidden); logits = lm_head(hidden)`; skipping it
        # feeds an unnormalized residual stream to the unembedding, inflating logits
        # by ~2 orders of magnitude and cross-entropy into the thousands.
        final_norm = None
        if self.config.successor_ip is None and arch_info.final_norm_prefix:
            with torch.device('meta'):
                final_norm = ModelRegistry.build_final_norm(model_config, architecture)

        # Create the learned absolute position embedding (rank 0 only). GPT-Neo adds
        # wte + wpe; RoPE architectures leave this None and inject position per block.
        position_embedding = None
        if self.config.device_rank == 0 and arch_info.position_embedding_key:
            hidden_size = ModelRegistry.get_hidden_size(model_config, architecture)
            max_pos = getattr(model_config, 'max_position_embeddings', 2048)
            with torch.device('meta'):
                position_embedding = nn.Embedding(max_pos, hidden_size)


        gc.collect()
        
        print(f"Lazy loading weights from {safetensors_path}")
        
        # Check if we need to load from sharded files
        import os
        import glob
        
        if not os.path.exists(safetensors_path):
            # Look for sharded files like model-00001-of-00002.safetensors
            model_dir = os.path.dirname(safetensors_path)
            shard_pattern = os.path.join(model_dir, "model-*.safetensors")
            shard_files = sorted(glob.glob(shard_pattern))
            
            if shard_files:
                print(f"Found {len(shard_files)} sharded safetensors files")
                safetensors_files = shard_files
            else:
                raise FileNotFoundError(f"No safetensors file found at {safetensors_path} or sharded files")
        else:
            safetensors_files = [safetensors_path]
        
        # Build a key → shard_file index by opening each shard ONCE
        # This avoids re-opening shard files for every single key lookup
        key_to_shard = {}
        for shard_file in safetensors_files:
            with safe_open(shard_file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    key_to_shard[key] = shard_file
        

        
        # Force all weights to fp32 regardless of stored dtype
        target_dtype = torch.float32
        
        # Open all shard files ONCE and keep handles alive to avoid repeated mmap of the entire file
        shard_handles = {}
        for shard_file in safetensors_files:
            shard_handles[shard_file] = safe_open(shard_file, framework="pt", device="cpu")
        
        # Helper: load a single tensor from the correct shard (using pre-opened handle)
        def get_tensor(key):
            shard_file = key_to_shard.get(key)
            if shard_file is None:
                return None
            f = shard_handles[shard_file]
            tensor = f.get_tensor(key)
            if target_dtype is not None:
                tensor = tensor.to(target_dtype)
            return tensor
        
        # Load layer weights one key at a time to minimise peak memory
        try:
            for i, layer_idx in enumerate(self.config.assigned_layers):
                print(f"Loading layer {layer_idx} weights...")
                layer_prefix = f"{arch_info.layer_prefix}.{layer_idx}."
                
                # Collect matching keys for this layer
                layer_keys = [k for k in key_to_shard if k.startswith(layer_prefix)]
                assert len(layer_keys) > 0, f"No weights found for layer {layer_idx} with prefix {layer_prefix}"
                print(f"   Found {len(layer_keys)} weight keys for layer {layer_idx}")
                
                # Load one tensor at a time straight into a state dict, then load
                layer_state_dict = {}
                for key in layer_keys:
                    tensor_key = key[len(layer_prefix):]
                    tensor = get_tensor(key)
                    if tensor is not None:
                        layer_state_dict[tensor_key] = tensor
                
                # assign=True is NOT enough when moving from meta to a real device if the destination
                # device expects materialization. 
                # Instead, we materialize the shell first with to_empty(), then load weights.
                # This ensures we have real storage allocated.
                layers[i].to_empty(device=self.device)
                layers[i].load_state_dict(layer_state_dict, strict=False, assign=True)
                
                # Move to device (redundant if already on device, but needed if loaded from CPU tensor)
                layers[i].to(self.device)

                
                # Free temporary state dict and collect garbage between layers
                del layer_state_dict
                gc.collect()
        except Exception as e:
            print(f"❌ ERROR during layer weight loading: {e}")
            import traceback
            traceback.print_exc()
            raise
            

        
        
        # Load embedding layer weights
        if embedding_layer is not None:
            print(f"Loading embedding weights...")
            embedding_key = arch_info.embedding_key
            
            if embedding_key in key_to_shard:
                embedding_tensor = get_tensor(embedding_key)
                if embedding_tensor is not None:
                    # Materialize empty shell first
                    embedding_layer.to_empty(device=self.device)
                    embedding_layer.load_state_dict({"weight": embedding_tensor}, assign=True)
                    
                    # Move to device
                    embedding_layer.to(self.device)

                    del embedding_tensor
                    gc.collect()
            else:
                print(f"Warning: Embedding key {embedding_key} not found, using random init")
            

        
        # Load LM head weights
        if lm_head is not None:
            try:
                print(f"Loading LM head weights...")
                lm_head_key = arch_info.lm_head_key
                
                if lm_head_key in key_to_shard:
                    lm_head_tensor = get_tensor(lm_head_key)
                    if lm_head_tensor is not None:
                        lm_head.load_state_dict({"weight": lm_head_tensor}, assign=True)
                        del lm_head_tensor
                        gc.collect()
                else:
                    print(f"Warning: LM head key {lm_head_key} not found, using random init")
                
                # Checks if LM head is still on meta device (meaning weights weren't loaded via assign=True)
                if getattr(lm_head, 'weight', None) is not None and lm_head.weight.device.type == 'meta':
                    # Materialize empty shell
                    lm_head.to_empty(device=self.device)
                    
                    if lm_head_key not in key_to_shard:
                        # Random init if no weights found and we just materialized empty shell
                        print(f"Initializing LM head with random weights")
                        lm_head.reset_parameters()
                        
                # Ensure it ends up on the right device (in case it was loaded to CPU above)
                lm_head.to(self.device)
                
                print(f"✅ LM head processed successfully")
            except Exception as e:
                print(f"❌ ERROR loading LM head: {e}")
                import traceback
                traceback.print_exc()
                raise

        # Load final-norm weights. Materialize first, then identity-init (weight 1,
        # bias 0) so a missing key degrades to a no-op norm rather than the garbage
        # that to_empty() leaves behind; real weights then overwrite via assign.
        if final_norm is not None:
            fn_prefix = arch_info.final_norm_prefix
            print(f"Loading final norm weights ({fn_prefix})...")
            final_norm.to_empty(device=self.device)
            with torch.no_grad():
                for pname, param in final_norm.named_parameters():
                    param.fill_(1.0 if pname.endswith("weight") else 0.0)

            fn_state = {}
            for suffix in ("weight", "bias"):
                key = f"{fn_prefix}.{suffix}"
                if key in key_to_shard:
                    tensor = get_tensor(key)
                    if tensor is not None:
                        fn_state[suffix] = tensor

            if fn_state:
                final_norm.load_state_dict(fn_state, strict=False, assign=True)
                del fn_state
                gc.collect()
            else:
                print(f"⚠️  Final norm {fn_prefix} not found in checkpoint — "
                      f"using identity norm (logit scale will be wrong)")

            final_norm.to(self.device)
            # Frozen base-model parameters: only LoRA adapters train.
            for param in final_norm.parameters():
                param.requires_grad_(False)
            print(f"✅ Final norm processed successfully")

        # Load learned position-embedding weights (GPT-Neo style, rank 0 only)
        if position_embedding is not None:
            pos_key = arch_info.position_embedding_key
            print(f"Loading position embedding weights ({pos_key})...")
            if pos_key in key_to_shard:
                pos_tensor = get_tensor(pos_key)
                if pos_tensor is not None:
                    position_embedding.to_empty(device=self.device)
                    position_embedding.load_state_dict({"weight": pos_tensor}, assign=True)
                    del pos_tensor
                    gc.collect()
            else:
                print(f"⚠️  Position embedding {pos_key} not found — zero-initializing "
                      f"(model will have no positional signal)")
                position_embedding.to_empty(device=self.device)
                with torch.no_grad():
                    position_embedding.weight.zero_()

            position_embedding.to(self.device)
            for param in position_embedding.parameters():
                param.requires_grad_(False)
            print(f"✅ Position embedding processed successfully")

        # Close all open shard handles and free the shard index
        for h in shard_handles.values():
            del h
        del shard_handles
        del key_to_shard
        gc.collect()

        # Create rotary embedding for position embeddings computation
        # Qwen2/LLaMA/Mistral layers require position_embeddings to be passed
        if self.needs_position_ids:
            if self.architecture == ModelArchitecture.LLAMA:
                from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
                self.rotary_emb = LlamaRotaryEmbedding(config=model_config, device=self.device)
                print(f"Created LlamaRotaryEmbedding for position embeddings")
            elif self.architecture == ModelArchitecture.MISTRAL:
                from transformers.models.mistral.modeling_mistral import MistralRotaryEmbedding  
                self.rotary_emb = MistralRotaryEmbedding(config=model_config, device=self.device)
                print(f"Created MistralRotaryEmbedding for position embeddings")
            elif self.architecture == ModelArchitecture.QWEN2:
                from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
                self.rotary_emb = Qwen2RotaryEmbedding(config=model_config, device=self.device)
                print(f"Created Qwen2RotaryEmbedding for position embeddings")
            elif self.architecture == ModelArchitecture.PHI:
                from transformers.models.phi.modeling_phi import PhiRotaryEmbedding
                self.rotary_emb = PhiRotaryEmbedding(config=model_config, device=self.device)
                print(f"Created PhiRotaryEmbedding for position embeddings")
            else:
                self.rotary_emb = None
        else:
            self.rotary_emb = None
        
        return layers, embedding_layer, lm_head, final_norm, position_embedding

    def forward_step_local(self, micro_batch_id: int, batch: Dict[str, torch.Tensor]):
        assert isinstance(micro_batch_id, int)
        assert isinstance(batch, dict)
        assert 'input_ids' in batch
        assert 'attention_mask' in batch
        assert self.config.device_rank == 0, "Local forward step only for rank 0"
        assert self.embedding_layer is not None, "Embedding layer not loaded for rank 0"
        
        self._cleanup_old_cache_entries()
        
        with self.cache_lock:
            assert micro_batch_id not in self.forward_cache, f"Batch {micro_batch_id} already in cache"
        
        #GPU CHANGE
        input_ids = batch['input_ids'].to(self.device)

        # Padded positions are excluded from the loss (data_loader pads to 256).
        labels = build_masked_labels(input_ids, batch.get('attention_mask'))

        # Reference embedding composition: hidden = wte(ids) + wpe(pos).
        # position_embedding is None for RoPE architectures, which apply position
        # inside each block instead.
        hidden_states = self.embedding_layer(input_ids)
        if self.position_embedding is not None:
            seq_len = input_ids.size(1)
            position_ids = torch.arange(
                seq_len, dtype=torch.long, device=self.device
            ).unsqueeze(0).expand(input_ids.size(0), -1)
            hidden_states = hidden_states + self.position_embedding(position_ids)

        hidden_states = hidden_states.detach()
        hidden_states.requires_grad_(True)
        
        with self.cache_lock:
            self.forward_cache[micro_batch_id] = {
                'input': hidden_states,
                'labels': labels.detach().clone()
            }
        
        output_tensor = hidden_states

        # Compute and pass position embeddings for RoPE models
        if self.needs_position_ids:
            batch_size, seq_len, _ = hidden_states.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Compute position embeddings with CORRECT head_dim shape
            head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
            dummy_value = torch.zeros(
                batch_size, seq_len, head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            cos, sin = self.rotary_emb(dummy_value, position_ids)
            position_embeddings = (cos, sin)
            
            position_embeddings = (cos, sin)
            
            for i, layer in enumerate(self.layers):


                
                layer_output = layer(output_tensor, position_ids=position_ids, position_embeddings=position_embeddings)
                
                # CRITICAL FIX: Handle both tuple and tensor returns
                if isinstance(layer_output, tuple):
                    output_tensor = layer_output[0]
                else:
                    # Layer returned tensor directly, use it as-is
                    output_tensor = layer_output
                

        else:
            # For GPT-Neo and other models that don't use position_ids at block level
            for layer in self.layers:
                output_tensor = layer(output_tensor)[0]

        with self.cache_lock:
            self.forward_cache[micro_batch_id]['output'] = output_tensor
            
            if not output_tensor.requires_grad:
                output_tensor.requires_grad_(True)

        if self.config.successor_ip is None:
            print(f"Final stage processing micro-batch {micro_batch_id}")
            
            assert self.lm_head is not None, "LM head not loaded for final stage"
            
            with self.cache_lock:
                labels = self.forward_cache[micro_batch_id]['labels']

            # Final normalization before the unembedding, as in the reference
            # forward. Gradients still flow back to output_tensor through it.
            normed_states = (self.final_norm(output_tensor)
                             if self.final_norm is not None else output_tensor)
            logits = self.lm_head(normed_states)

            # Causal LM label shifting: predict token[i+1] from token[i].
            # logits[:, :-1] aligns with labels[:, 1:] (next-token targets).
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

            print(f"Loss for micro-batch {micro_batch_id}: {loss.item():.4f}")

            initial_gradient = torch.autograd.grad(loss, output_tensor, retain_graph=False)[0]
            self.backward_step(micro_batch_id, initial_gradient, loss.item())
        else:
            print(f"Forward pass complete for micro-batch {micro_batch_id}, sending to {self.config.successor_ip}")
            
            send_tensor = output_tensor.detach()
            compressed_payload = self.compression_engine.compress_tensor(send_tensor, tensor_type='activations')
            compressed_labels = self.compression_engine.compress_tensor(labels.detach(), tensor_type='labels')
            
            metadata = {
                "micro_batch_id": micro_batch_id,
                "labels": base64.b64encode(compressed_labels).decode('utf-8'),
                "send_timestamp": time.time()
            }
            
            tensor_message = Message(
                message_type=MessageType.TENSOR,
                payload=compressed_payload,
                metadata=metadata
            )
            
            send_start = time.time()
            bytes_sent = self.network_manager.send_message(
                self.config.successor_ip,
                29500,
                tensor_message
            )
            send_latency = time.time() - send_start
            mb_size = bytes_sent / (1024 * 1024)
            speed_mbps = mb_size / send_latency if send_latency > 0 else 0
            print(f"   🚀 Sent {mb_size:.2f} MB in {send_latency:.4f}s ({speed_mbps:.2f} MB/s) to {self.config.successor_ip}")

    def forward_step_remote(self, micro_batch_id: int, hidden_states: torch.Tensor, labels_data: dict = None):
        assert isinstance(micro_batch_id, int)
        assert isinstance(hidden_states, torch.Tensor)
        assert self.config.device_rank > 0, "Remote forward step only for rank > 0"
        
        self._cleanup_old_cache_entries()
        
        with self.cache_lock:
            assert micro_batch_id not in self.forward_cache, f"Batch {micro_batch_id} already in cache"
        
        # GPU CHANGE
        hidden_states = hidden_states.to(self.device)

        hidden_states = hidden_states.detach()
        hidden_states.requires_grad_(True)
        
        cache_entry = {
            'input': hidden_states
        }
        
        if labels_data is not None:
            cache_entry.update(labels_data)
        
        with self.cache_lock:
            self.forward_cache[micro_batch_id] = cache_entry
        
        output_tensor = hidden_states

        # Compute and pass position embeddings for RoPE models
        if self.needs_position_ids:
            batch_size, seq_len, _ = hidden_states.shape
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            # Compute position embeddings with CORRECT head_dim shape
            head_dim = self.model_config.hidden_size // self.model_config.num_attention_heads
            dummy_value = torch.zeros(
                batch_size, seq_len, head_dim,
                dtype=hidden_states.dtype,
                device=hidden_states.device
            )
            cos, sin = self.rotary_emb(dummy_value, position_ids)
            position_embeddings = (cos, sin)
            
            position_embeddings = (cos, sin)
            
            for i, layer in enumerate(self.layers):


                
                layer_output = layer(output_tensor, position_ids=position_ids, position_embeddings=position_embeddings)
                
                # CRITICAL FIX: Handle both tuple and tensor returns
                if isinstance(layer_output, tuple):
                    output_tensor = layer_output[0]
                else:
                    # Layer returned tensor directly, use it as-is
                    output_tensor = layer_output
                

        else:
            # For GPT-Neo and other models that don't use position_ids at block level
            for layer in self.layers:
                output_tensor = layer(output_tensor)[0]

        with self.cache_lock:
            self.forward_cache[micro_batch_id]['output'] = output_tensor
            
            if not output_tensor.requires_grad:
                output_tensor.requires_grad_(True)

        if self.config.successor_ip is None:
            print(f"Final stage processing micro-batch {micro_batch_id}")
            
            assert self.lm_head is not None, "LM head not loaded for final stage"
            
            with self.cache_lock:
                labels = self.forward_cache[micro_batch_id].get('labels')
            assert labels is not None, "Labels not found in final stage"

            labels = labels.to(self.device)

            # Final normalization before the unembedding, as in the reference
            # forward. Gradients still flow back to output_tensor through it.
            normed_states = (self.final_norm(output_tensor)
                             if self.final_norm is not None else output_tensor)
            logits = self.lm_head(normed_states)

            # Causal LM label shifting: predict token[i+1] from token[i].
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )

            print(f"Loss for micro-batch {micro_batch_id}: {loss.item():.4f}")

            initial_gradient = torch.autograd.grad(loss, output_tensor, retain_graph=False)[0]
            self.backward_step(micro_batch_id, initial_gradient, loss.item())
        else:
            print(f"Forward pass complete for micro-batch {micro_batch_id}, sending to {self.config.successor_ip}")
            
            send_tensor = output_tensor.detach()
            compressed_payload = self.compression_engine.compress_tensor(send_tensor, tensor_type='activations')
            
            metadata = {
                "micro_batch_id": micro_batch_id,
                "send_timestamp": time.time()
            }
            
            with self.cache_lock:
                if 'labels_b64_str' in self.forward_cache[micro_batch_id]:
                    metadata["labels"] = self.forward_cache[micro_batch_id]['labels_b64_str']
            
            tensor_message = Message(
                message_type=MessageType.TENSOR,
                payload=compressed_payload,
                metadata=metadata
            )
            
            send_start = time.time()
            bytes_sent = self.network_manager.send_message(
                self.config.successor_ip,
                29500,
                tensor_message
            )
            send_latency = time.time() - send_start
            mb_size = bytes_sent / (1024 * 1024)
            speed_mbps = mb_size / send_latency if send_latency > 0 else 0
            print(f"   🚀 Sent {mb_size:.2f} MB in {send_latency:.4f}s ({speed_mbps:.2f} MB/s) to {self.config.successor_ip}")

    def backward_step(self, micro_batch_id: int, gradient: torch.Tensor, loss_value: float = None):
        assert isinstance(micro_batch_id, int)
        assert isinstance(gradient, torch.Tensor)
        
        with self.cache_lock:
            assert micro_batch_id in self.forward_cache, f"Batch {micro_batch_id} not in cache"
            cached_data = self.forward_cache[micro_batch_id]
            input_tensor = cached_data['input']

        # Move gradient to device (fp32 everywhere)
        gradient = gradient.to(device=self.device)
        
        self.lora_manager.zero_grad()

        try:
            
            output_tensor = cached_data['output']
            output_tensor.backward(gradient=gradient, retain_graph=False)
            input_gradient = input_tensor.grad

            if self.config.predecessor_ip is not None:
                assert input_gradient is not None, "No input gradient computed"
                        
            step_successful = self.lora_manager.step()
            
            if step_successful:
                stats = self.lora_manager.get_learning_stats()
                print(f"✅ LoRA updated batch {micro_batch_id}: "
                    f"grad_norm={stats['average_gradient_norm']:.6f}, "
                    f"param_norm={stats['average_parameter_norm']:.6f}")
            else:
                print(f"⚠️  No LoRA optimizer for batch {micro_batch_id}")

        except Exception as e:
            print(f"Error in backward computation for batch {micro_batch_id}: {e}")
            raise
        finally:
            with self.cache_lock:
                if micro_batch_id in self.forward_cache:
                    del self.forward_cache[micro_batch_id]
            
            # Clear GPU cache
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        if self.config.predecessor_ip is not None:
            print(f"Backward pass complete for micro-batch {micro_batch_id}, sending gradient to {self.config.predecessor_ip}")
            
            compressed_payload = self.compression_engine.compress_tensor(input_gradient.detach(), tensor_type='gradients')
            
            metadata = {
                "micro_batch_id": micro_batch_id,
                "send_timestamp": time.time()
            }
            if loss_value is not None:
                metadata["loss_value"] = loss_value
            
            gradient_message = Message(
                message_type=MessageType.GRADIENT,
                payload=compressed_payload,
                metadata=metadata
            )
            send_start = time.time()
            bytes_sent = self.network_manager.send_message(
                self.config.predecessor_ip,
                29500,
                gradient_message
            )
            send_latency = time.time() - send_start
            mb_size = bytes_sent / (1024 * 1024)
            speed_mbps = mb_size / send_latency if send_latency > 0 else 0
            print(f"   � Sent {mb_size:.2f} MB in {send_latency:.4f}s ({speed_mbps:.2f} MB/s) to {self.config.predecessor_ip}")
        else:
            print(f"✅ LoRA parameters updated for coordinator, micro-batch {micro_batch_id}")

    def get_lora_state_dict(self):
        # Pass a plain list so the manager can iterate and match adapter.layer_block
        return self.lora_manager.get_lora_state_dict(
            list(self.layers), self.config.assigned_layers
        )

    @property
    def target_modules(self) -> list:
        """The target module names used when LoRA was applied — for adapter_config.json."""
        return getattr(self, "_target_modules", [])

    def set_lora_state_dict(self, state_dict):
        return self.lora_manager.set_lora_state_dict(list(self.layers), state_dict)