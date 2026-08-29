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

class PartitionInfeasible(RuntimeError):
    """A partition strategy could not place every layer within memory limits."""


def _usable_mem_gb(dev, utilization_limit, is_master, embedding_size_gb):
    if dev.stats.device_type == "cuda":
        u = dev.stats.memory_gb * utilization_limit
    else:
        # match partition_model: coordinator (RAM-4)*0.65, worker (RAM-5)*0.60
        u = max(0.0, dev.stats.memory_gb - (4.0 if is_master else 5.0)) * (0.65 if is_master else 0.60)
    if is_master:
        u = max(0.0, u - embedding_size_gb)
    return u


def _assert_feasible(assignments, devices, layer_size_gb, embedding_size_gb,
                     master_ip, utilization_limit):
    for d in devices:
        need = assignments.get(d.ip, 0) * layer_size_gb
        have = _usable_mem_gb(d, utilization_limit, d.ip == master_ip, embedding_size_gb)
        if need > have + 1e-9:
            raise PartitionInfeasible(
                f"{d.ip}: needs {need:.3f} GB for {assignments[d.ip]} layers, "
                f"has {have:.3f} GB usable")


def _smart_assignments(sorted_devices, num_layers, layer_size_gb, embedding_size_gb,
                       master_ip, utilization_limit):
    """Compute memory-aware layer assignments for smart strategy."""
    coordinator = sorted_devices[0]
    assert coordinator.ip == master_ip, "Coordinator must be first device"

    # Estimate embedding size
    vocab_size = 50257
    hidden_size = 768  # Default, overridden by actual config in partition_model
    embedding_size_gb_computed = (vocab_size * hidden_size * 4) / (1024 ** 3)

    # Minimum memory: 1 layer + embedding
    min_coordinator_memory_gb = layer_size_gb + embedding_size_gb_computed

    if coordinator.stats.device_type == 'cuda':
        coordinator_usable = coordinator.stats.memory_gb * utilization_limit
    else:
        coordinator_usable = max(0, coordinator.stats.memory_gb - 4.0) * 0.65

    if coordinator_usable < min_coordinator_memory_gb:
        print(f"\n❌ COORDINATOR TOO WEAK TO START TRAINING")
        print(f"   Coordinator ({master_ip}): {coordinator.stats.memory_gb:.2f} GB {coordinator.stats.device_type.upper()}")
        print(f"   Usable after overhead: {coordinator_usable:.2f} GB")
        print(f"   Minimum required: {min_coordinator_memory_gb:.2f} GB")
        print(f"   Shortfall: {min_coordinator_memory_gb - coordinator_usable:.2f} GB")
        print(f"\n   💡 SOLUTION: Upgrade coordinator memory or use a more powerful device as coordinator")
        sys.exit(1)

    # Calculate Capacity & Assignments
    assignments = {}
    remaining_layers = num_layers

    for device in sorted_devices:
        if remaining_layers > 0:
            assignments[device.ip] = 1
            remaining_layers -= 1
        else:
            assignments[device.ip] = 0

    # Distribute remaining layers by usable memory capacity
    for device in sorted_devices:
        if remaining_layers <= 0:
            break

        if device.ip == master_ip:
            continue

        if device.stats.device_type == 'cuda':
            usable_mem = device.stats.memory_gb * utilization_limit
        else:
            usable_mem = max(0, device.stats.memory_gb - 5.0) * 0.60

        already_used = assignments[device.ip] * layer_size_gb
        capacity = int((usable_mem - already_used) / layer_size_gb)
        to_take = min(max(0, capacity), remaining_layers)

        assignments[device.ip] += to_take
        remaining_layers -= to_take

    # Smart overflow handling
    if remaining_layers > 0:
        for device in sorted_devices:
            if remaining_layers <= 0:
                break

            if device.stats.device_type == 'cuda':
                usable_mem = device.stats.memory_gb * utilization_limit
            else:
                usable_mem = max(0, device.stats.memory_gb - 5.0) * 0.60

            current_usage = assignments[device.ip] * layer_size_gb
            spare_capacity_gb = usable_mem - current_usage
            additional_layers = int(spare_capacity_gb / layer_size_gb)

            if additional_layers > 0:
                to_add = min(additional_layers, remaining_layers)
                assignments[device.ip] += to_add
                remaining_layers -= to_add

        if remaining_layers > 0:
            raise PartitionInfeasible(
                f"Unable to fit {remaining_layers} layers on cluster")

    # Reserve LM head memory on last worker
    active_for_lm_check = [d for d in sorted_devices if assignments[d.ip] > 0]
    if len(active_for_lm_check) > 1:
        last_device = active_for_lm_check[-1]
        lm_head_size_gb = (vocab_size * hidden_size * 4) / (1024 ** 3)

        if last_device.stats.device_type == 'cuda':
            last_usable = last_device.stats.memory_gb * utilization_limit
        else:
            last_usable = max(0, last_device.stats.memory_gb - 5.0) * 0.60

        last_needed = assignments[last_device.ip] * layer_size_gb + lm_head_size_gb
        if last_needed > last_usable:
            overflow = last_needed - last_usable
            layers_to_shed = max(1, int(overflow / layer_size_gb) + 1)
            shed = min(layers_to_shed, assignments[last_device.ip] - 1)
            if shed > 0:
                assignments[last_device.ip] -= shed
                remaining_layers += shed

                # Try to place shed layers on other devices
                for device in sorted_devices:
                    if remaining_layers <= 0:
                        break
                    if device.ip == last_device.ip:
                        continue
                    if device.stats.device_type == 'cuda':
                        u = device.stats.memory_gb * utilization_limit
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

    return assignments


def compute_assignments(strategy, devices, num_layers, layer_size_gb,
                        embedding_size_gb, master_ip, utilization_limit, seed=0):
    healthy = [d for d in devices if d.status == DeviceStatus.HEALTHY]
    assert healthy, "no healthy devices"
    ordered = sorted(healthy, key=lambda d: (0 if d.ip == master_ip else 1, -d.stats.flops))
    workers = [d for d in ordered if d.ip != master_ip]
    assert workers, "need at least one worker"

    if strategy == "smart":
        return _smart_assignments(ordered, num_layers, layer_size_gb,
                                  embedding_size_gb, master_ip, utilization_limit)

    a = {d.ip: 0 for d in ordered}
    coord_layer = 1 if num_layers > len(workers) else 0
    a[master_ip] = coord_layer
    rem = num_layers - coord_layer

    if strategy == "round_robin":
        i = 0
        while rem > 0:
            a[workers[i % len(workers)].ip] += 1
            rem -= 1
            i += 1
    elif strategy == "proportional":
        weights = [max(d.stats.flops, 1e-6) for d in workers]
        total = sum(weights)
        raw = [rem * w / total for w in weights]
        base = [int(x) for x in raw]
        for d, b in zip(workers, base):
            a[d.ip] = b
        leftover = rem - sum(base)
        for d, _ in sorted(zip(workers, raw), key=lambda p: p[1] - int(p[1]),
                           reverse=True)[:leftover]:
            a[d.ip] += 1
        for d in workers:
            if a[d.ip] == 0:
                donor = max(workers, key=lambda x: a[x.ip])
                a[donor.ip] -= 1
                a[d.ip] += 1
    elif strategy == "random":
        rng = __import__("random").Random(seed)
        for d in workers:
            a[d.ip] += 1
        rem -= len(workers)
        for _ in range(max(0, rem)):
            a[rng.choice(workers).ip] += 1
    else:
        raise ValueError(f"unknown strategy {strategy!r}")

    assert sum(a.values()) == num_layers, (a, num_layers)
    _assert_feasible(a, ordered, layer_size_gb, embedding_size_gb,
                     master_ip, utilization_limit)
    return a


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

    def partition_model(self, master_ip: str, strategy: str = "smart") -> dict[str, PipelineConfig]:
        print(f"\n✅ Computing {strategy.upper()} Partition (Master: {master_ip})...")

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

        # Estimate embedding size (coordinator is always rank 0, loads embedding + may load LM head)
        vocab_size = getattr(config, 'vocab_size', 50257)
        hidden_size = ModelRegistry.get_hidden_size(config, architecture)
        embedding_size_gb = (vocab_size * hidden_size * 4) / (1024 ** 3)  # float32 = 4 bytes
        print(f"   Embedding size: ~{embedding_size_gb:.2f} GB (float32)")

        # Compute assignments using the specified strategy
        assignments = compute_assignments(
            strategy, list(self.devices.values()), num_layers, layer_size_gb,
            embedding_size_gb, master_ip, self.utilization_limit,
            seed=self.model_config.get("seed", 0))
        sorted_devices = sorted(
            [d for d in self.devices.values() if d.status == DeviceStatus.HEALTHY],
            key=lambda d: (0 if d.ip == master_ip else 1, -d.stats.flops))

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
    configs = manager.partition_model("127.0.0.1")

    print("\n✅ Generated configurations:")
    for ip, config in configs.items():
        print(f"  Device {ip}: {len(config.assigned_layers)} layers, Rank {config.device_rank}")

    # Basic Validations
    total_assigned = sum(len(c.assigned_layers) for c in configs.values())
    assert total_assigned == 32, f"❌ Total layers {total_assigned} != 32"

    print("✅ All tests passed successfully")
