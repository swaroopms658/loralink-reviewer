import argparse
import sys
import time
import json
import torch
import json
import struct
import base64
from pipeline_engine import PipelineStage
from device_manager import DeviceManager, PipelineConfig
from network_protocol import NetworkManager, Message, MessageType
import benchmarking
import data_loader
from transformers import AutoTokenizer
import gc
# from safetensors.torch import safe_open
# import tempfile
# import shutil
from typing import Optional
import queue
import threading
import os
import io
import datetime


class TeeLogger:
    """Duplicates stdout to both console and a log file."""
    def __init__(self, log_dir="."):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, f"coordinator_log_{timestamp}.txt")
        self.terminal = sys.stdout
        self.log_file = open(self.log_path, "w", encoding="utf-8", buffering=1)
    
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()
        sys.stdout = self.terminal


DTYPE_MAP = {
    "F16": torch.float16, "F32": torch.float32, "F64": torch.float64,
    "I8": torch.int8, "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "U8": torch.uint8, "BOOL": torch.bool, "BF16": torch.bfloat16,
}

SAVE_DTYPE_MAP = {v: k for k, v in DTYPE_MAP.items()}

GRADIENT_QUEUE = queue.Queue()
PIPELINE_ENGINE: Optional[PipelineStage] = None
NETWORK_MANAGER: Optional[NetworkManager] = None
DEVICE_MANAGER: Optional[DeviceManager] = None
LORA_WEIGHTS_CACHE = {}
ALL_WEIGHTS_RECEIVED = threading.Event()
READY_WORKERS = set()
ALL_WORKERS_READY = threading.Event()
ACTIVE_WORKER_IPS = []  # Track which workers received configs (excluding idle devices)
SELF_HOST_IP = None  # this process's own --host-ip; stamped into outbound messages so
                     # the receiver identifies the true sender (loopback accept() reports
                     # 127.0.0.1 for every 127.0.0.x client, collapsing worker identity)

# def stream_and_merge_tensors(shard_path, lora_weights, lora_config):

#     print(f"    [GEN] Opening {os.path.basename(shard_path)} for streaming...")
#     with safe_open(shard_path, framework="pt", device="cpu") as f:
#         keys = list(f.keys())
#         for i, key in enumerate(keys):
#             try:
                
#                 tensor = f.get_tensor(key)
#                 merged_tensor, was_merged = apply_lora_if_needed(
#                     key, tensor, lora_weights, lora_config
#                 )
#                 yield key, merged_tensor, was_merged
#             except Exception as e:
#                 print(f"      [GEN] CRITICAL ERROR while processing key '{key}': {e}")
#                 raise
#     print("    [GEN] Finished streaming all keys.")

# def save_file_streamed(tensor_generator, filename: str):
#     header = {}
#     current_offset = 0
#     tensor_count = 0
#     merged_count = 0  

#     try:
#         with tempfile.TemporaryFile() as f_data:
#             print("  Pass 1/2: Streaming tensors to temporary disk storage...")
            
#             for key, tensor, was_merged in tensor_generator:
#                 tensor_count += 1
#                 if was_merged:
#                     merged_count += 1 

#                 tensor_bytes = tensor.cpu().numpy().tobytes()
#                 data_len = len(tensor_bytes)
#                 header[key] = {
                    
                    
#                     "dtype": SAVE_DTYPE_MAP[tensor.dtype],
#                     "shape": list(tensor.shape),
#                     "data_offsets": [current_offset, current_offset + data_len],
#                 }
#                 current_offset += data_len
#                 f_data.write(tensor_bytes)
            
#             print(f"    [SAVE] Successfully wrote {tensor_count} tensors to temporary file.")

            

#             header_json = json.dumps({"__metadata__": {"format": "pt"}, **header}).encode("utf-8")
#             header_len = len(header_json)

#             print(f"  Pass 2/2: Assembling final file from header and temporary data...")
#             f_data.seek(0)
#             with open(filename, "wb") as f_final:
#                 print("    [SAVE] Writing header...")
#                 f_final.write(struct.pack("<Q", header_len))
#                 f_final.write(header_json)
#                 print("    [SAVE] Copying tensor data...")
#                 shutil.copyfileobj(f_data, f_final)
#         print("  Streaming save complete.")
#     except Exception as e:
#         print(f"  CRITICAL ERROR during file save: {e}")
#         raise
    
    
#     return tensor_count, merged_count

def _coordinator_message_handler(sender_ip: str, message: Message):
    global DEVICE_MANAGER, PIPELINE_ENGINE, GRADIENT_QUEUE, LORA_WEIGHTS_CACHE, ALL_WEIGHTS_RECEIVED
    global READY_WORKERS, ALL_WORKERS_READY, ACTIVE_WORKER_IPS

    try:
        # Prefer the sender's self-reported IP: on loopback, accept() sees every
        # 127.0.0.x client as 127.0.0.1, so sender_ip alone can't tell workers apart.
        src_ip = message.metadata.get("src_ip") or sender_ip

        if message.message_type == MessageType.BENCHMARK_RESULT:
            assert DEVICE_MANAGER is not None, "DeviceManager not initialized"
            print(f"Received benchmark result from {src_ip}")
            DEVICE_MANAGER.process_benchmark_result(src_ip, message.payload)

        elif message.message_type == MessageType.WORKER_READY:
            print(f"✅ Worker {src_ip} is READY")
            READY_WORKERS.add(src_ip)
            
            # Check if all ACTIVE workers are ready (idle devices were excluded)
            if len(ACTIVE_WORKER_IPS) > 0:
                if len(READY_WORKERS) >= len(ACTIVE_WORKER_IPS):
                    print(f"✅ All {len(ACTIVE_WORKER_IPS)} active workers are ready!")
                    ALL_WORKERS_READY.set()

        elif message.message_type == MessageType.GRADIENT:
            print(f"[Network Thread] Received gradient from {sender_ip}")
            assert PIPELINE_ENGINE is not None, "PipelineEngine not initialized for gradient"
            
            try:
                gradient = PIPELINE_ENGINE.compression_engine.decompress_tensor(message.payload)
                micro_batch_id = message.metadata.get("micro_batch_id")
                loss_value = message.metadata.get("loss_value")
                
                # Compute end-to-end communication latency
                send_timestamp = message.metadata.get("send_timestamp")
                if send_timestamp is not None:
                    e2e_latency = time.time() - send_timestamp
                    if e2e_latency < 0: e2e_latency = 0.0  # Handle clock skew
                    print(f"📡 Gradient receive latency from {sender_ip}: {e2e_latency:.6f}s (end-to-end)")
                
                if micro_batch_id is None:
                    print(f"Error: No micro_batch_id in gradient message from {sender_ip}")
                    return
                
                GRADIENT_QUEUE.put((micro_batch_id, gradient, loss_value))
                print(f"Successfully queued gradient for batch {micro_batch_id}")
                
            except Exception as e:
                print(f"Error processing gradient from {sender_ip}: {e}")

        elif message.message_type == MessageType.LORA_WEIGHTS_RESPONSE:
            print(f"Received LoRA weights from {src_ip}")

            try:
                buffer = io.BytesIO(message.payload)
                worker_weights = torch.load(buffer, map_location='cpu')
                LORA_WEIGHTS_CACHE[src_ip] = worker_weights
                
                expected_workers = len([ip for ip in DEVICE_MANAGER.devices.keys() 
                                      if ip != list(DEVICE_MANAGER.devices.keys())[0]])
                
                if len(LORA_WEIGHTS_CACHE) >= expected_workers:
                    print(f"Received weights from all {expected_workers} workers")
                    ALL_WEIGHTS_RECEIVED.set()
                    
            except Exception as e:
                print(f"Error processing LoRA weights from {sender_ip}: {e}")
                
    except Exception as e:
        print(f"Error in coordinator message handler for {sender_ip}: {e}")
def _worker_message_handler(sender_ip: str, message: Message):
    global PIPELINE_ENGINE, NETWORK_MANAGER

    try:
        if message.message_type == MessageType.RUN_BENCHMARK:
            print(f"Received benchmark command from {sender_ip}")
            try:
                benchmark_results = benchmarking.run_benchmark()
                result_payload = json.dumps(benchmark_results).encode('utf-8')
                coordinator_ip = message.metadata.get("coordinator_ip")
                
                if coordinator_ip is None:
                    print("Error: No coordinator IP in benchmark message")
                    return

                result_message = Message(
                    message_type=MessageType.BENCHMARK_RESULT,
                    payload=result_payload,
                    metadata={"src_ip": SELF_HOST_IP}
                )

                NETWORK_MANAGER.send_message(coordinator_ip, 29500, result_message)
                print(f"Sent benchmark results to coordinator {coordinator_ip}")
                
            except Exception as e:
                print(f"Error running benchmark: {e}")

        elif message.message_type == MessageType.SET_CONFIG:
            print(f"Received configuration from {sender_ip}")
            try:
                config_data = json.loads(message.payload.decode('utf-8'))
                config = PipelineConfig(
                    assigned_layers=config_data["assigned_layers"],
                    lora_rank=config_data["lora_rank"],
                    predecessor_ip=config_data["predecessor_ip"],
                    successor_ip=config_data["successor_ip"],
                    master_ip=config_data["master_ip"],
                    world_size=config_data["world_size"],
                    device_rank=config_data["device_rank"],
                    model_name=config_data["model_name"]
                )

                PIPELINE_ENGINE = PipelineStage(config, NETWORK_MANAGER)
                print(f"Worker configured with {len(config.assigned_layers)} layers, rank {config.device_rank}")
                
                # Send WORKER_READY signal to coordinator
                ready_message = Message(
                    message_type=MessageType.WORKER_READY,
                    payload=b"",
                    metadata={"rank": config.device_rank, "src_ip": SELF_HOST_IP}
                )
                NETWORK_MANAGER.send_message(config.master_ip, 29500, ready_message)
                print(f"✅ Sent WORKER_READY signal to coordinator {config.master_ip}")
                
            except Exception as e:
                print(f"Error setting config: {e}")

        elif message.message_type == MessageType.TENSOR:
            if PIPELINE_ENGINE is None:
                print("Error: PipelineEngine not initialized for tensor")
                return
                
            try:
                micro_batch_id = message.metadata.get("micro_batch_id")
                if micro_batch_id is None:
                    print("Error: No micro_batch_id in tensor message")
                    return
                
                tensor = PIPELINE_ENGINE.compression_engine.decompress_tensor(message.payload)
                
                # Compute end-to-end communication latency
                send_timestamp = message.metadata.get("send_timestamp")
                if send_timestamp is not None:
                    e2e_latency = time.time() - send_timestamp
                    if e2e_latency < 0: e2e_latency = 0.0
                    print(f"📡 Tensor receive latency from {sender_ip}: {e2e_latency:.6f}s (end-to-end)")
                
                
                labels_data = None
                if "labels" in message.metadata:
                    try:
                        compressed_labels_b64 = message.metadata["labels"]
                        compressed_labels = base64.b64decode(compressed_labels_b64)
                        labels = PIPELINE_ENGINE.compression_engine.decompress_tensor(compressed_labels)
                        labels_data = {
                            'labels': labels,
                            'labels_b64_str': compressed_labels_b64
                        }
                    except Exception as e:
                        print(f"Error processing labels: {e}")
                
                
                PIPELINE_ENGINE.forward_step_remote(micro_batch_id, tensor, labels_data)
                
            except Exception as e:
                print(f"Error processing tensor: {e}")

        elif message.message_type == MessageType.GRADIENT:
            if PIPELINE_ENGINE is None:
                print("Error: PipelineEngine not initialized for gradient")
                return
                
            try:
                micro_batch_id = message.metadata.get("micro_batch_id")
                if micro_batch_id is None:
                    print("Error: No micro_batch_id in gradient message")
                    return
                    
                gradient = PIPELINE_ENGINE.compression_engine.decompress_tensor(message.payload)
                loss_value = message.metadata.get("loss_value")
                
                # Compute end-to-end communication latency
                send_timestamp = message.metadata.get("send_timestamp")
                if send_timestamp is not None:
                    e2e_latency = time.time() - send_timestamp
                    if e2e_latency < 0: e2e_latency = 0.0
                    print(f"📡 Gradient receive latency from {sender_ip}: {e2e_latency:.6f}s (end-to-end)")
                
                PIPELINE_ENGINE.backward_step(micro_batch_id, gradient, loss_value)
                
            except Exception as e:
                print(f"Error processing gradient: {e}")

        elif message.message_type == MessageType.GET_LORA_WEIGHTS:
            print(f"Received request for LoRA weights from {sender_ip}")
            if PIPELINE_ENGINE is None:
                print("Error: PipelineEngine not initialized for LoRA weights")
                return
                
            try:
                lora_weights = PIPELINE_ENGINE.get_lora_state_dict()
                
                if not lora_weights:
                    print("Warning: No LoRA weights found - model may not have been trained")
                else:
                    print(f"Collected {len(lora_weights)} LoRA parameter tensors")
                
                buffer = io.BytesIO()
                torch.save(lora_weights, buffer)
                weights_payload = buffer.getvalue()
                
                response_message = Message(
                    message_type=MessageType.LORA_WEIGHTS_RESPONSE,
                    payload=weights_payload,
                    metadata={"num_parameters": len(lora_weights), "src_ip": SELF_HOST_IP}
                )
                
                # --- FIX: Send to Master, NOT Sender ---
                # The sender_ip might be 127.0.0.1 (proxy), so we ignore it.
                # We send the weights explicitly to the Master IP we stored in config.
                target_ip = PIPELINE_ENGINE.config.master_ip
                
                print(f"Sent LoRA weights to Master at {target_ip} ({len(weights_payload)} bytes)")
                NETWORK_MANAGER.send_message(target_ip, 29500, response_message)
                # ---------------------------------------
                
            except Exception as e:
                print(f"Error sending LoRA weights: {e}")

        elif message.message_type == MessageType.SHUTDOWN:
            shutdown_reason = message.payload.decode('utf-8')
            print(f"\n🚨 Received SHUTDOWN signal from coordinator")
            print(f"   Reason: {shutdown_reason}")
            print(f"   This worker was excluded from training (0 layers assigned)")
            print(f"   Shutting down gracefully...")
            sys.exit(0)  # Graceful exit
                
    except Exception as e:
        print(f"Error in worker message handler for {sender_ip}: {e}")


def save_lora_adapters(
    all_lora_weights: dict,
    lora_rank: int,
    output_path: str = "./lora_adapters",
    base_model_path: str = None,
    target_modules: list = None,
):
    import os
    import json

    os.makedirs(output_path, exist_ok=True)

    adapter_model_path = os.path.join(output_path, "adapter_model.bin")
    torch.save(all_lora_weights, adapter_model_path)
    print(f"Saved {len(all_lora_weights)} LoRA parameters to {adapter_model_path}")

    file_size_mb = os.path.getsize(adapter_model_path) / (1024 * 1024)
    print(f"Adapter file size: {file_size_mb:.2f} MB")

    # Guarantee a valid base path — never use a placeholder
    if base_model_path is None:
        base_model_path = "<base_model_path>"

    # Default target_modules when not provided
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Standard PEFT adapter_config.json schema:
    # https://huggingface.co/docs/peft/package_reference/lora
    adapter_config = {
        "base_model_name_or_path": base_model_path,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        # PEFT expects 'r', not 'lora_rank'
        "r": lora_rank,
        "lora_alpha": 16.0,
        "lora_dropout": 0.0,
        "target_modules": target_modules,
        "modules_to_save": None,
        "fan_in_fan_out": False,
        "bias": "none",
    }

    config_path = os.path.join(output_path, "adapter_config.json")
    with open(config_path, "w") as f:
        json.dump(adapter_config, f, indent=2)
    print(f"Saved adapter config to {config_path}")
    print(f"  base_model_name_or_path : {base_model_path}")
    print(f"  target_modules          : {target_modules}")
    print(f"  r (rank)                : {lora_rank}")

def _summary_path(p):
    """Sibling path for the summary CSV: foo/x.csv -> foo/x.summary.csv."""
    return (p[:-4] + ".summary.csv") if p.endswith(".csv") else (p + ".summary.csv")

def run_coordinator(args):
    global NETWORK_MANAGER, DEVICE_MANAGER, PIPELINE_ENGINE, SELF_HOST_IP

    worker_ips = args.workers.split(',')
    assert len(worker_ips) > 0, "❌ No worker IPs provided"

    coordinator_ip = args.host_ip
    SELF_HOST_IP = args.host_ip
    all_device_ips = [coordinator_ip] + worker_ips
    print(f"✅ Starting coordinator for cluster: {all_device_ips}")
    
    # Activate TeeLogger: duplicate all print output to a timestamped log file
    tee_logger = TeeLogger(log_dir=".")
    sys.stdout = tee_logger
    print(f"📝 Logging coordinator output to: {tee_logger.log_path}")

    # Reconfigure logging to use the TeeLogger (sys.stdout) so libraries using logging
    # (like compression_engine) also print to the log file.
    import logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)


    NETWORK_MANAGER = NetworkManager(args.host_ip or "0.0.0.0", 29500, _coordinator_message_handler)
    NETWORK_MANAGER.start_server()

    # Use model_name from args
    model_config = {"model_name": args.model_path, "seed": args.seed}
    DEVICE_MANAGER = DeviceManager(all_device_ips, model_config)

    remote_worker_ips = worker_ips
    print("✅ Starting benchmarking phase for remote workers...")
    for ip in remote_worker_ips:
        benchmark_message = Message(message_type=MessageType.RUN_BENCHMARK, payload=b"", metadata={"coordinator_ip": coordinator_ip})
        NETWORK_MANAGER.send_message(ip, 29500, benchmark_message)
        print(f"✅ Sent benchmark command to {ip}")

    print("✅ Running local benchmark for coordinator...")
    local_stats = benchmarking.run_benchmark()
    local_payload = json.dumps(local_stats).encode('utf-8')
    DEVICE_MANAGER.process_benchmark_result(coordinator_ip, local_payload)

    timeout = 30
    start_time = time.time()
    while time.time() - start_time < timeout:
        healthy_count = sum(1 for device in DEVICE_MANAGER.devices.values()
                          if device.status.name == "HEALTHY")
        if healthy_count == len(all_device_ips):
            print(f"✅ All {len(all_device_ips)} devices completed benchmarking")
            break
        time.sleep(1)
    else:
        print("❌ Timeout waiting for benchmark results")
        failed_devices = [ip for ip, dev in DEVICE_MANAGER.devices.items() if dev.status.name != "HEALTHY"]
        print(f"❌ Failed to get results from: {failed_devices}")
        sys.exit(1)

    print("✅ Computing model partition...")
    configs = DEVICE_MANAGER.partition_model(args.host_ip, strategy=args.partition_strategy)

    print("✅ Sending configurations to workers...")
    
    # Identify idle workers (those not in configs)
    all_worker_ips = [ip for ip in all_device_ips if ip != coordinator_ip]
    active_worker_ips_from_configs = [ip for ip in configs.keys() if ip != coordinator_ip]
    idle_worker_ips = [ip for ip in all_worker_ips if ip not in active_worker_ips_from_configs]
    
    # Send shutdown to idle workers
    if idle_worker_ips:
        print(f"🚨 Sending shutdown signal to {len(idle_worker_ips)} idle worker(s): {idle_worker_ips}")
        for idle_ip in idle_worker_ips:
            shutdown_message = Message(
                message_type=MessageType.SHUTDOWN,
                payload=b"No layers assigned, shutting down gracefully",
                metadata={"reason": "idle"}
            )
            NETWORK_MANAGER.send_message(idle_ip, 29500, shutdown_message)
            print(f"   Sent shutdown to {idle_ip}")
    
    # Send configs to active workers
    for ip, config in configs.items():
        if ip == coordinator_ip:
            continue

        config_data = {
            "assigned_layers": config.assigned_layers,
            "lora_rank": config.lora_rank,
            "predecessor_ip": config.predecessor_ip,
            "successor_ip": config.successor_ip,
            "master_ip": config.master_ip,
            "world_size": config.world_size,
            "device_rank": config.device_rank,
            "model_name": config.model_name
        }

        config_message = Message(
            message_type=MessageType.SET_CONFIG,
            payload=json.dumps(config_data).encode('utf-8'),
            metadata={}
        )

        NETWORK_MANAGER.send_message(ip, 29500, config_message)

    # Set ACTIVE_WORKER_IPS BEFORE initializing coordinator pipeline
    # This ensures the message handler knows the expected worker count
    # before any WORKER_READY signals arrive during coordinator loading
    global ACTIVE_WORKER_IPS
    ACTIVE_WORKER_IPS = [ip for ip in configs.keys() if ip != coordinator_ip]
    num_active_workers = len(ACTIVE_WORKER_IPS)

    coordinator_config = configs.get(coordinator_ip)
    assert coordinator_config is not None, "❌ No coordinator config found"
    print(f"✅ Initializing coordinator pipeline with {len(coordinator_config.assigned_layers)} layers")
    PIPELINE_ENGINE = PipelineStage(coordinator_config, NETWORK_MANAGER)

    # Wait for all workers to signal they're ready
    # IMPORTANT: Only wait for workers that received configs (active workers)
    # Idle devices were filtered out and never received SET_CONFIG
    
    if num_active_workers > 0:
        print(f"⏳ Waiting for {num_active_workers} active workers to be ready...")
        if num_active_workers < len(remote_worker_ips):
            idle_count = len(remote_worker_ips) - num_active_workers
            print(f"   Note: {idle_count} idle worker(s) were excluded and won't send ready signals")
        
        workers_ready = ALL_WORKERS_READY.wait(timeout=120)
        if not workers_ready:
            print("❌ Timeout waiting for workers to be ready")
            ready_list = list(READY_WORKERS)
            missing = [ip for ip in ACTIVE_WORKER_IPS if ip not in ready_list]
            print(f"❌ Ready workers: {ready_list}")
            print(f"❌ Missing workers: {missing}")
            sys.exit(1)
        print(f"✅ All workers confirmed ready, proceeding with training")

    print("✅ Loading tokenizer and dataset...")
    # Use dynamic model path from args
    model_path = f"./models/{args.model_path}"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except Exception:
        print("⚠️ Fast tokenizer failed, falling back to slow tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = data_loader.get_data_loader(
        tokenizer, num_samples=args.num_samples, dataset_name=args.dataset,
        split="train", eval_holdout=args.eval_holdout)

    from loralink_reviewer_response.metrics_logger import (
        append_rows, RUN_COLUMNS, SUMMARY_COLUMNS)
    import time as _time
    _run_start = _time.perf_counter()
    _losses, _latencies = [], []
    _sim = "loopback" if (args.host_ip or "").startswith("127.") else "real"
    global_batch = 0

    print("✅ Starting training with real data...")
    for epoch in range(args.epochs):
        for batch in train_loader:
            print(f"✅ Processing batch {epoch}:{global_batch}")
            t0 = _time.perf_counter()
            PIPELINE_ENGINE.forward_step_local(global_batch, batch)
            try:
                rid, gradient, loss_value = GRADIENT_QUEUE.get(timeout=300)
            except queue.Empty:
                print(f"❌ Timeout waiting for gradient for batch {global_batch}")
                sys.exit(1)
            assert rid == global_batch, "❌ Mismatched batch ID!"
            PIPELINE_ENGINE.backward_step(rid, gradient)
            step_latency = _time.perf_counter() - t0
            if loss_value is not None:
                _losses.append(float(loss_value))
                print(f"✅ loss[{epoch}:{global_batch}] = {loss_value:.4f}")
            _latencies.append(step_latency)

            if args.metrics_csv:
                cs = PIPELINE_ENGINE.compression_engine.get_compression_stats()
                ratio = cs.get("average_compression_ratio", "").rstrip("x") or ""
                append_rows(args.metrics_csv, [{
                    "run_tag": args.run_tag, "seed": args.seed,
                    "strategy": args.partition_strategy,
                    "compression": os.environ.get("LORALINK_LOSSY_COMPRESSION", "1"),
                    "dataset": args.dataset, "model": args.model_path,
                    "n_workers": len(ACTIVE_WORKER_IPS), "epoch": epoch,
                    "global_batch": global_batch,
                    "loss": "" if loss_value is None else float(loss_value),
                    "step_latency_s": step_latency, "comp_ratio": ratio,
                    "bytes_sent": PIPELINE_ENGINE.compression_engine.stats["total_compressed_bytes"],
                    "bytes_saved": (PIPELINE_ENGINE.compression_engine.stats["total_original_bytes"]
                                    - PIPELINE_ENGINE.compression_engine.stats["total_compressed_bytes"]),
                    "sim": _sim, "timestamp": _time.time(),
                }], RUN_COLUMNS)
            global_batch += 1

    if args.metrics_csv:
        st = PIPELINE_ENGINE.compression_engine.stats
        counts = [len(c.assigned_layers) for c in configs.values()]
        import statistics as _stx
        append_rows(_summary_path(args.metrics_csv), [{
            "run_tag": args.run_tag, "seed": args.seed,
            "strategy": args.partition_strategy,
            "compression": os.environ.get("LORALINK_LOSSY_COMPRESSION", "1"),
            "dataset": args.dataset, "model": args.model_path,
            "n_workers": len(ACTIVE_WORKER_IPS), "sim": _sim,
            "n_batches": global_batch,
            "mean_loss": (sum(_losses)/len(_losses)) if _losses else "",
            "last_loss": _losses[-1] if _losses else "",
            "mean_step_latency_s": (sum(_latencies)/len(_latencies)) if _latencies else "",
            "total_bytes_sent": st["total_compressed_bytes"],
            "total_bytes_saved": st["total_original_bytes"] - st["total_compressed_bytes"],
            "overall_comp_ratio": (st["total_original_bytes"] / st["total_compressed_bytes"])
                                   if st["total_compressed_bytes"] else "",
            "wall_time_s": _time.perf_counter() - _run_start,
            "partition_map": ";".join(f"{ip}:{len(c.assigned_layers)}"
                                      for ip, c in configs.items()),
            "partition_balance_std": _stx.pstdev(counts) if len(counts) > 1 else 0.0,
        }], SUMMARY_COLUMNS)

    print("✅ Training complete. Starting model reconstruction...")
    
    for ip in remote_worker_ips:
        get_weights_message = Message(message_type=MessageType.GET_LORA_WEIGHTS, payload=b"", metadata={})
        NETWORK_MANAGER.send_message(ip, 29500, get_weights_message)
        print(f"Requested LoRA weights from {ip}")
    
    print("Waiting to receive LoRA weights from all workers...")
    event_was_set = ALL_WEIGHTS_RECEIVED.wait(timeout=600)
    assert event_was_set, "Timeout waiting for LoRA weights from workers"
    
    # print("STREAMING model reconstruction...")

    
    # coordinator_lora_weights = PIPELINE_ENGINE.get_lora_state_dict()
    # print(f"Collected coordinator LoRA weights: {len(coordinator_lora_weights)} parameters")

    
    # del PIPELINE_ENGINE
    # PIPELINE_ENGINE = None
    # gc.collect()
    # print("Cleared training model from memory")

    
    # run_streaming_reconstruction(
    #     coordinator_weights=coordinator_lora_weights,
    #     worker_weights_cache=LORA_WEIGHTS_CACHE,
    #     coordinator_config=coordinator_config,
    #     base_model_path="./models/EleutherAI/gpt-neo-2.7B",
    #     output_path="./finetuned_model"
    # )
    
    # print("Model reconstruction COMPLETE")
    # print("Coordinator shutting down...")
    # NETWORK_MANAGER.stop_server()

    print("Saving LoRA adapters...")
    
    coordinator_lora_weights = PIPELINE_ENGINE.get_lora_state_dict()
    print(f"Collected coordinator LoRA weights: {len(coordinator_lora_weights)} parameters")

    all_lora_weights = coordinator_lora_weights.copy()
    for worker_ip, weights in LORA_WEIGHTS_CACHE.items():
        print(f"Merging weights from worker {worker_ip}: {len(weights)} parameters")
        all_lora_weights.update(weights)

    print(f"Total LoRA parameters collected: {len(all_lora_weights)}")

    # Use dynamic model path for adapter config
    model_path = f"./models/{args.model_path}"
    save_lora_adapters(
        all_lora_weights=all_lora_weights,
        lora_rank=coordinator_config.lora_rank,
        output_path="./lora_adapters",
        base_model_path=model_path,
        target_modules=PIPELINE_ENGINE.target_modules,
    )

    print("LoRA adapters saved to ./lora_adapters")

# def run_streaming_reconstruction(coordinator_weights, worker_weights_cache, coordinator_config, 
#                                base_model_path, output_path):

#     print("=== DEBUGGING LoRA RECONSTRUCTION ===")
    
    
#     all_lora_weights = coordinator_weights.copy()
#     for worker_ip, weights in worker_weights_cache.items():
#         print(f"Merging weights from worker {worker_ip}: {len(weights)} parameters")
        
#         conflicts = set(all_lora_weights.keys()) & set(weights.keys())
#         if conflicts:
#             print(f"WARNING: {len(conflicts)} overlapping keys with worker {worker_ip}")
#             for key in list(conflicts)[:5]:  
#                 print(f"  Conflict: {key}")
        
        
#         for key, tensor in weights.items():
#             if key not in all_lora_weights:
#                 all_lora_weights[key] = tensor
#             else:
#                 print(f"  Skipping duplicate key: {key}")
        
#     print(f"\nTotal LoRA weights to merge: {len(all_lora_weights)} parameters")
    
    
#     print("\n=== ANALYZING LoRA KEY PATTERNS ===")
#     lora_key_patterns = {}
#     for key in all_lora_weights.keys():
#         if '.lora_A' in key:
#             base_key = key.replace('.lora_A', '')
#             lora_key_patterns[base_key] = lora_key_patterns.get(base_key, []) + ['A']
#         elif '.lora_B' in key:
#             base_key = key.replace('.lora_B', '')
#             lora_key_patterns[base_key] = lora_key_patterns.get(base_key, []) + ['B']
#         elif 'lora_A' in key:  
#             base_key = key.replace('lora_A', 'weight')
#             lora_key_patterns[base_key] = lora_key_patterns.get(base_key, []) + ['A_alt']
#         elif 'lora_B' in key:
#             base_key = key.replace('lora_B', 'weight')
#             lora_key_patterns[base_key] = lora_key_patterns.get(base_key, []) + ['B_alt']
    
#     print(f"Found {len(lora_key_patterns)} potential LoRA pairs:")
#     for base_key, components in list(lora_key_patterns.items())[:10]:
#         print(f"  {base_key} -> {components}")
    
    
#     safetensors_files = []
#     model_file = os.path.join(base_model_path, "model.safetensors")
#     if os.path.exists(model_file):
#         safetensors_files = [model_file]
#         print(f"Found single model file: {model_file}")
#     else:
#         index_file = os.path.join(base_model_path, "model.safetensors.index.json")
#         if os.path.exists(index_file):
#             with open(index_file, 'r') as f:
#                 index_data = json.load(f)
#             shard_files = set(index_data["weight_map"].values())
#             safetensors_files = [os.path.join(base_model_path, f) for f in shard_files]
#             print(f"Found {len(safetensors_files)} sharded model files")
    
#     if not safetensors_files:
#         print("ERROR: No model files found!")
#         return
    
#     os.makedirs(output_path, exist_ok=True)
    
#     lora_config = {
#         'rank': coordinator_config.lora_rank,
#         'alpha': 16.0
#     }
#     print(f"LoRA config: rank={lora_config['rank']}, alpha={lora_config['alpha']}")
    
    
#     total_merged_weights = 0
#     total_processed_weights = 0
    
#     for i, shard_path in enumerate(safetensors_files):
#         output_file = f"model-{i+1:05d}-of-{len(safetensors_files):05d}.safetensors"
#         output_path_full = os.path.join(output_path, output_file)

#         print(f"\n=== LAZY PROCESSING {os.path.basename(shard_path)} -> {output_file} ===")

        
#         tensor_generator = stream_and_merge_tensors(shard_path, all_lora_weights, lora_config)

        
#         processed_in_shard, merged_in_shard = save_file_streamed(tensor_generator, output_path_full)
#         total_processed_weights += processed_in_shard
#         total_merged_weights += merged_in_shard

#         print(f"Saved {output_file} ({merged_in_shard} weights merged)")
    
#     print(f"\n=== FINAL SUMMARY ===")
#     print(f"Total weights processed: {total_processed_weights}")
#     print(f"Total weights merged: {total_merged_weights}")
#     print(f"Merge success rate: {(total_merged_weights/max(total_processed_weights,1))*100:.2f}%")
    
#     if total_merged_weights == 0:
#         print("WARNING: NO WEIGHTS WERE MERGED! Check key naming patterns.")
#         print("Debug: Let's check exact key format...")
#         if all_lora_weights:
#             sample_lora_key = list(all_lora_weights.keys())[0]
#             print(f"Sample LoRA key: '{sample_lora_key}'")
        
        
#         with safe_open(safetensors_files[0], framework="pt", device="cpu") as f:
#             sample_model_key = list(f.keys())[0]
#             print(f"Sample model key: '{sample_model_key}'")
    
    
#     auxiliary_files = ["config.json", "generation_config.json", "tokenizer.json", 
#                       "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"]
    
#     for filename in auxiliary_files:
#         src_path = os.path.join(base_model_path, filename)
#         if os.path.exists(src_path):
#             dst_path = os.path.join(output_path, filename)
#             shutil.copy2(src_path, dst_path)
    
#     if len(safetensors_files) > 1:
#         create_model_index(safetensors_files, output_path)
    
#     print("Streaming reconstruction complete")
    
# def apply_lora_if_needed(weight_key, original_tensor, lora_weights, lora_config):

#     if not weight_key.endswith('.weight'):
#         return original_tensor, False
    
#     base_key = weight_key[:-7]  
    
#     if base_key.startswith('transformer.h.'):
        
#         parts = base_key.split('.')
#         if len(parts) >= 3 and parts[0] == 'transformer' and parts[1] == 'h':
            
#             layer_num = parts[2]
#             rest_path = '.'.join(parts[3:])
#             lora_base_key = f"{layer_num}.{rest_path}"
#         else:
#             return original_tensor, False
#     else:
#         return original_tensor, False

#     lora_a_key = f"{lora_base_key}.lora_A"
#     lora_b_key = f"{lora_base_key}.lora_B"
    
#     if lora_a_key in lora_weights and lora_b_key in lora_weights:
#         print(f"    MERGING: {weight_key} <- {lora_a_key} + {lora_b_key}")
        
#         lora_a = lora_weights[lora_a_key]
#         lora_b = lora_weights[lora_b_key]
        
#         if lora_b.shape[1] != lora_a.shape[0]:
#             print(f"    ERROR: Dimension mismatch - B:{lora_b.shape} A:{lora_a.shape}")
#             return original_tensor, False
        
#         if lora_a.shape[1] != original_tensor.shape[1] or lora_b.shape[0] != original_tensor.shape[0]:
#             print(f"    ERROR: Size mismatch - Original:{original_tensor.shape} vs LoRA:{lora_b.shape}x{lora_a.shape}")
#             return original_tensor, False

#         lora_delta = torch.matmul(lora_b, lora_a)

#         scaling = lora_config['alpha'] / lora_config['rank']
#         delta_scaled = (lora_delta * scaling).to(original_tensor.dtype)
#         original_tensor.add_(delta_scaled)
#         original_tensor.requires_grad = False

#         print(f"    SUCCESS (in-place): Applied scaling {scaling:.2f}, delta norm: {torch.norm(lora_delta).item():.6f}")
#         return original_tensor, True

#     return original_tensor, False

# def create_model_index(safetensors_files, output_path):
#     weight_map = {}
    
#     for i, shard_path in enumerate(safetensors_files):
#         shard_name = f"model-{i+1:05d}-of-{len(safetensors_files):05d}.safetensors"
#         output_shard = os.path.join(output_path, shard_name)
        
#         with safe_open(output_shard, framework="pt", device="cpu") as f:
#             for key in f.keys():
#                 weight_map[key] = shard_name
    
#     index_data = {
#         "metadata": {"format": "pt"},
#         "weight_map": weight_map
#     }
    
#     index_path = os.path.join(output_path, "model.safetensors.index.json")
#     with open(index_path, 'w') as f:
#         json.dump(index_data, f, indent=2)
    
#     print(f"Created model index with {len(weight_map)} weights")

def run_worker(args):
    global NETWORK_MANAGER, SELF_HOST_IP

    SELF_HOST_IP = args.host_ip
    print("✅ Starting worker node...")
    NETWORK_MANAGER = NetworkManager(args.host_ip or "0.0.0.0", 29500, _worker_message_handler)
    NETWORK_MANAGER.start_server()

    print("✅ Worker ready, waiting for commands...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("✅ Worker shutting down...")
        NETWORK_MANAGER.stop_server()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed LoRA Training System")
    parser.add_argument("--role", choices=["coordinator", "worker"], required=True,
                      help="Role of this process")
    parser.add_argument("--workers", type=str,
                      help="Comma-separated list of worker IPs (coordinator only)")
    parser.add_argument("--host-ip", type=str,
                      help="The LAN IP address of this machine; used as the "
                           "NetworkManager bind address for both coordinator and worker roles")
    parser.add_argument("--model-path", type=str, default="EleutherAI/gpt-neo-2.7B",
                      help="Model name or path (relative to ./models/) - e.g., 'EleutherAI/gpt-neo-2.7B' or 'meta-llama/Llama-2-7b-hf'")
    parser.add_argument("--dataset", type=str, default="wikitext",
                      choices=["wikitext", "dolly", "e2e"],
                      help="Dataset to use for training (default: wikitext)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-holdout", type=int, default=0)
    parser.add_argument("--partition-strategy",
                        choices=["smart", "round_robin", "proportional", "random"],
                        default="smart")
    parser.add_argument("--base-model", type=str, default=None,
                        help="alias for --model-path; wins if both set")
    parser.add_argument("--run-tag", type=str, default="")
    parser.add_argument("--metrics-csv", type=str, default="")

    args = parser.parse_args()

    if args.base_model:
        args.model_path = args.base_model

    import random as _random, numpy as _np
    _random.seed(args.seed); _np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"🎲 seed={args.seed}")

    # In-process network shim: when LORALINK_NET_SHIM="delay_ms,loss_pct" is set,
    # monkeypatch NetworkManager.send_message to add latency / drop sends. The shim
    # models packet loss as connection failure -- a dropped send raises
    # ConnectionError; there is no retry path, so the coordinator treats it as a
    # failed step. network_protocol.py itself stays untouched.
    _shim_spec = os.environ.get("LORALINK_NET_SHIM")
    if _shim_spec:
        import network_protocol as _np_mod, random as _shim_rand, time as _shim_time
        _d_ms, _loss = _shim_spec.split(",")
        _shim_delay = float(_d_ms) / 1000.0
        _shim_loss = float(_loss) / 100.0
        _shim_orig = _np_mod.NetworkManager.send_message
        def _shim_send(peer_ip, peer_port, message):
            if _shim_delay:
                _shim_time.sleep(_shim_delay)
            if _shim_loss and _shim_rand.random() < _shim_loss:
                raise ConnectionError("net-shim simulated drop")
            return _shim_orig(peer_ip, peer_port, message)
        _np_mod.NetworkManager.send_message = staticmethod(_shim_send)
        print(f"🌐 net-shim active: delay={_d_ms}ms loss={_loss}%")

    if args.role == "coordinator":
        assert args.workers is not None, "❌ --workers required for coordinator role"
        run_coordinator(args)
    elif args.role == "worker":
        run_worker(args)
    else:
        print("❌ Invalid role specified")
        sys.exit(1)