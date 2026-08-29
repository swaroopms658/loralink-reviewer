import enum
import dataclasses
import time
import json
import typing
import sys
from transformers import AutoConfig
from network_protocol import Message, MessageType
from model_registry import ModelRegistry, ModelArchitecture

class DeviceStatus(enum.Enum):
    PENDING = enum.auto()
    BENCHMARKING = enum.auto()
    HEALTHY = enum.auto()
    FAILED = enum.auto()

@dataclasses.dataclass
class DeviceStats:
    flops: float = 0.0
    memory_gb: float = 0.0
    device_type: str = "cpu"  # Added: To distinguish utilization rules

@dataclasses.dataclass
class DeviceHandle:
    ip: str
    status: DeviceStatus = dataclasses.field(default=DeviceStatus.PENDING)
    stats: DeviceStats = dataclasses.field(default_factory=DeviceStats)

@dataclasses.dataclass
class PipelineConfig:
    assigned_layers: list[int]
    lora_rank: int
    predecessor_ip: typing.Optional[str]
    successor_ip: typing.Optional[str]
    master_ip: str
    world_size: int
    device_rank: int
    model_name: str = "EleutherAI/gpt-neo-2.7B"  # Added: Useful for workers to know what to load

class DeviceManager:
    def __init__(self, worker_ips: list[str], model_config: dict, utilization_limit: float = 0.70):
        assert isinstance(worker_ips, list)
        assert all(isinstance(ip, str) for ip in worker_ips)
        assert isinstance(model_config, dict)
        
        self.model_config = model_config
        self.utilization_limit = utilization_limit  # GPU Memory Limit (70%)
        self.devices: dict[str, DeviceHandle] = {}
        
        for ip in worker_ips:
            self.devices[ip] = DeviceHandle(ip=ip)
    
    def start_benchmarking(self, coordinator_ip: str, worker_port: int, network_manager):
        assert isinstance(coordinator_ip, str)
        assert isinstance(worker_port, int)
        
        for ip, device in self.devices.items():
            if device.status == DeviceStatus.PENDING:
                device.status = DeviceStatus.BENCHMARKING
                
                benchmark_message = Message(
                    message_type=MessageType.RUN_BENCHMARK,
                    payload=b"",
                    metadata={"coordinator_ip": coordinator_ip}
                )
                
                network_manager.send_message(ip, worker_port, benchmark_message)
                print(f"✅ Sent benchmark command to {ip}")
    
    def process_benchmark_result(self, worker_ip: str, result_payload: bytes):
        assert isinstance(worker_ip, str)
        assert isinstance(result_payload, bytes)
        assert worker_ip in self.devices, f"❌ Unknown worker IP: {worker_ip}"
        
        result_data = json.loads(result_payload.decode("utf-8"))
        
        # Validation
        assert "flops" in result_data
        assert "memory_gb" in result_data
        assert result_data["flops"] > 0
        
        # Update Stats
        self.devices[worker_ip].stats.flops = float(result_data["flops"])
        self.devices[worker_ip].stats.memory_gb = float(result_data["memory_gb"])
        self.devices[worker_ip].stats.device_type = result_data.get("device_type", "cpu") # Default to cpu if missing
        self.devices[worker_ip].status = DeviceStatus.HEALTHY
        
        print(f"✅ Processed result for {worker_ip} [{self.devices[worker_ip].stats.device_type}]: "
              f"{result_data['flops']:.2f} TFLOPS, {result_data['memory_gb']:.2f} GB")
    
    def partition_model(self, master_ip: str) -> dict[str, PipelineConfig]:
        print(f"\n✅ Computing SMART Memory-Aware Partition (Master: {master_ip})...")
        
        # 1. Detect Architecture and Determine Model Size & Layer Size
        model_name = self.model_config.get("model_name", "EleutherAI/gpt-neo-2.7B")
        
        try:
            # Detect architecture using model registry
            architecture = ModelRegistry.detect_architecture(model_name)
            print(f"   Detected architecture: {architecture.value}")
            
            if architecture == ModelArchitecture.UNKNOWN:
                print("⚠️ Unknown architecture, using heuristics")
                raise ValueError("Unknown architecture")
            
            config = AutoConfig.from_pretrained(model_name)
            
            # Get architecture-aware layer count and size estimation
            num_layers = ModelRegistry.get_num_layers(config, architecture)
            size_info = ModelRegistry.estimate_model_size(config, architecture)
            layer_size_gb = size_info['per_layer_gb']
            
            print(f"   Model: {num_layers} layers")
            print(f"   Per Layer Size: ~{layer_size_gb:.4f} GB")
            
        except Exception as e:
            print(f"⚠️ Model detection failed ({e}), using fallback heuristics")
            num_layers = self.model_config.get("num_layers", 32)
            layer_size_gb = 0.5  # Conservative estimate
            architecture = ModelArchitecture.GPT_NEO

        healthy_devices = [d for d in self.devices.values() if d.status == DeviceStatus.HEALTHY]
        assert len(healthy_devices) > 0, "No healthy devices available"

        # 2. Sort devices: Force Master IP to be index 0 (Rank 0)
        sorted_devices = sorted(healthy_devices, key=lambda d: (0 if d.ip == master_ip else 1, -d.stats.flops))
        
        # 3. COORDINATOR VALIDATION: Ensure coordinator can hold at least 1 layer + embeddings
        coordinator = sorted_devices[0]
        assert coordinator.ip == master_ip, "Coordinator must be first device"
        
        # Estimate embedding size (coordinator is always rank 0, loads embedding + may load LM head)
        vocab_size = getattr(config, 'vocab_size', 50257)
        hidden_size = ModelRegistry.get_hidden_size(config, architecture)
        embedding_size_gb = (vocab_size * hidden_size * 4) / (1024 ** 3)  # float32 = 4 bytes
        print(f"   Embedding size: ~{embedding_size_gb:.2f} GB (float32)")
        
        # Minimum memory: 1 layer + embedding
        min_coordinator_memory_gb = layer_size_gb + embedding_size_gb
        
        if coordinator.stats.device_type == 'cuda':
            coordinator_usable = coordinator.stats.memory_gb * self.utilization_limit
        else:
            # CPU: deduct OS/Python/framework overhead (4 GB), then use 65%
            coordinator_usable = max(0, coordinator.stats.memory_gb - 4.0) * 0.65
        
        if coordinator_usable < min_coordinator_memory_gb:
            print(f"\n❌ COORDINATOR TOO WEAK TO START TRAINING")
            print(f"   Coordinator ({master_ip}): {coordinator.stats.memory_gb:.2f} GB {coordinator.stats.device_type.upper()}")
            print(f"   Usable after overhead: {coordinator_usable:.2f} GB")
            print(f"   Minimum required: {min_coordinator_memory_gb:.2f} GB")
            print(f"   Shortfall: {min_coordinator_memory_gb - coordinator_usable:.2f} GB")
            print(f"\n   💡 SOLUTION: Upgrade coordinator memory or use a more powerful device as coordinator")
            sys.exit(1)

        # 4. Calculate Capacity & Assignments
        # FAIRNESS: Pre-assign 1 layer to every device to ensure all participate
        assignments = {}
        remaining_layers = num_layers
        
        for device in sorted_devices:
            if remaining_layers > 0:
                assignments[device.ip] = 1
                remaining_layers -= 1
            else:
                assignments[device.ip] = 0
        
        print(f"   Pre-assigned 1 layer to {min(num_layers, len(sorted_devices))} device(s), {remaining_layers} remaining")
        
        # Distribute remaining layers by usable memory capacity
        # CPU coordinator is hard-capped at 1 layer (only gets the pre-assigned layer)
        for device in sorted_devices:
            if remaining_layers <= 0: break
            
            # Skip coordinator - it only gets the first layer from pre-assignment
            if device.ip == master_ip:
                print(f"   ⏭️ Skipping coordinator ({master_ip}) - capped at 1 layer")
                continue
            
            # Calculate usable memory based on device type
            if device.stats.device_type == 'cuda':
                usable_mem = device.stats.memory_gb * self.utilization_limit
            else:
                usable_mem = max(0, device.stats.memory_gb - 5.0) * 0.60
            
            # Subtract already-assigned layer
            already_used = assignments[device.ip] * layer_size_gb
            capacity = int((usable_mem - already_used) / layer_size_gb)
            to_take = min(max(0, capacity), remaining_layers)
                
            assignments[device.ip] += to_take
            remaining_layers -= to_take
            
        # 5. SMART OVERFLOW HANDLING
        if remaining_layers > 0:
            print(f"   ⚠️ {remaining_layers} layers remain unassigned. Attempting smart overflow...")
            
            # Try to fit overflow layers on devices with spare capacity
            for device in sorted_devices:
                if remaining_layers <= 0:
                    break
                
                # Check if device has additional capacity
                if device.stats.device_type == 'cuda':
                    usable_mem = device.stats.memory_gb * self.utilization_limit
                else:
                    usable_mem = max(0, device.stats.memory_gb - 5.0) * 0.60
                
                current_usage = assignments[device.ip] * layer_size_gb
                spare_capacity_gb = usable_mem - current_usage
                additional_layers = int(spare_capacity_gb / layer_size_gb)
                
                if additional_layers > 0:
                    to_add = min(additional_layers, remaining_layers)
                    assignments[device.ip] += to_add
                    remaining_layers -= to_add
                    print(f"   Added {to_add} overflow layers to {device.ip}")
            
            # If still have remaining layers, fail with detailed error
            if remaining_layers > 0:
                print(f"\n❌ INSUFFICIENT MEMORY FOR TRAINING")
                print(f"   Unable to fit {remaining_layers} layers on cluster")
                print(f"\n   Cluster memory breakdown:")
                for device in sorted_devices:
                    print(f"   - {device.ip}: {device.stats.memory_gb:.2f} GB {device.stats.device_type.upper()} "
                          f"(assigned {assignments[device.ip]} layers)")
                print(f"\n   Total layers required: {num_layers}")
                print(f"   Layers per device: ~{layer_size_gb:.4f} GB each")
                print(f"\n   💡 SOLUTION: Add more devices to cluster or use devices with more memory")
                sys.exit(1)

        # 6. Reserve LM head memory on last worker
        # The last device in the pipeline creates an LM head (vocab_size × hidden_size)
        # Size depends on device dtype: float32 (4 bytes) on GPU, float16 (2 bytes) on CPU
        
        # Find last active device (highest-rank worker)
        active_for_lm_check = [d for d in sorted_devices if assignments[d.ip] > 0]
        if len(active_for_lm_check) > 1:
            last_device = active_for_lm_check[-1]
            
            # LM head size: always fp32 now
            lm_head_size_gb = (vocab_size * hidden_size * 4) / (1024 ** 3)  # float32
            if last_device.stats.device_type == 'cuda':
                last_usable = last_device.stats.memory_gb * self.utilization_limit
            else:
                last_usable = max(0, last_device.stats.memory_gb - 5.0) * 0.60
            
            print(f"   LM head size: ~{lm_head_size_gb:.2f} GB (fp32)")
            
            last_needed = assignments[last_device.ip] * layer_size_gb + lm_head_size_gb
            if last_needed > last_usable:
                overflow = last_needed - last_usable
                layers_to_shed = max(1, int(overflow / layer_size_gb) + 1)
                shed = min(layers_to_shed, assignments[last_device.ip] - 1)
                if shed > 0:
                    assignments[last_device.ip] -= shed
                    remaining_layers += shed
                    print(f"   ⚠️ Reserved LM head memory on {last_device.ip}: shed {shed} layer(s)")
                    
                    # Try to place shed layers on other devices
                    for device in sorted_devices:
                        if remaining_layers <= 0:
                            break
                        if device.ip == last_device.ip:
                            continue
                        if device.stats.device_type == 'cuda':
                            u = device.stats.memory_gb * self.utilization_limit
                        else:
                            u = max(0, device.stats.memory_gb - 5.0) * 0.60
                        if device.ip == master_ip:
                            u = max(0, u - embedding_size_gb)
                        current_usage = assignments[device.ip] * layer_size_gb
                        spare = int((u - current_usage) / layer_size_gb)
                        if spare > 0:
                            to_add = min(spare, remaining_layers)
                            assignments[device.ip] += to_add
                            remaining_layers -= to_add
        
        # 7. Generate Configs and Filter Idle Devices
        configs = {}
        current_layer_start = 0
        
        # Filter out devices with 0 layers (idle devices)
        active_devices = [d for d in sorted_devices if assignments[d.ip] > 0]
        
        if len(active_devices) < len(sorted_devices):
            idle_devices = [d.ip for d in sorted_devices if assignments[d.ip] == 0]
            print(f"   ℹ️ Excluding {len(idle_devices)} idle device(s): {idle_devices}")
        
        for rank, device in enumerate(active_devices):
            count = assignments[device.ip]
            assigned_indices = list(range(current_layer_start, current_layer_start + count))
            current_layer_start += count
            
            # --- SMART LORA LOGIC ---
            if device.stats.memory_gb < 8:
                lora_rank = 4
            elif device.stats.memory_gb < 16:
                lora_rank = 8
            else:
                lora_rank = 16
            # ------------------------
            
            pred_ip = active_devices[rank - 1].ip if rank > 0 else None
            succ_ip = active_devices[rank + 1].ip if rank < len(active_devices) - 1 else None
            
            config = PipelineConfig(
                assigned_layers=assigned_indices,
                lora_rank=lora_rank,
                predecessor_ip=pred_ip,
                successor_ip=succ_ip,
                master_ip=master_ip,
                world_size=len(active_devices),
                device_rank=rank,
                model_name=model_name
            )
            configs[device.ip] = config
            print(f"   Rank {rank}: {device.ip} -> {len(assigned_indices)} Layers (LoRA r={lora_rank})")

        return configs

if __name__ == "__main__":
    WORKER_IPS = ["127.0.0.1", "127.0.0.2", "127.0.0.3"]
    # Added model_name to config to test auto-sizing
    MODEL_CONFIG = {"num_layers": 32, "model_name": "EleutherAI/gpt-neo-2.7B"}
    
    print("✅ Initializing DeviceManager...")
    manager = DeviceManager(WORKER_IPS, MODEL_CONFIG)
    
    print("✅ Simulating benchmark results...")
    
    # Worker 1: Powerful GPU (24GB VRAM, High FLOPS)
    worker_1_result = json.dumps({
        "flops": 1000.0, 
        "memory_gb": 24.0, 
        "device_type": "cuda"
    }).encode("utf-8")
    
    # Worker 2: Mid-range GPU (12GB VRAM)
    worker_2_result = json.dumps({
        "flops": 300.0, 
        "memory_gb": 12.0, 
        "device_type": "cuda"
    }).encode("utf-8")
    
    # Worker 3: CPU or Weak GPU (6GB RAM)
    worker_3_result = json.dumps({
        "flops": 100.0, # Lower flops for CPU
        "memory_gb": 6.0, 
        "device_type": "cpu"
    }).encode("utf-8")
    
    manager.process_benchmark_result("127.0.0.1", worker_1_result)
    manager.process_benchmark_result("127.0.0.2", worker_2_result)
    manager.process_benchmark_result("127.0.0.3", worker_3_result)
    
    print("✅ Computing model partition...")
    configs = manager.partition_model()
    
    print("\n✅ Generated configurations:")
    for ip, config in configs.items():
        print(f"  Device {ip}: {len(config.assigned_layers)} layers, Rank {config.device_rank}")

    # Basic Validations
    total_assigned = sum(len(c.assigned_layers) for c in configs.values())
    assert total_assigned == 32, f"❌ Total layers {total_assigned} != 32"
    
    print("✅ All tests passed successfully")