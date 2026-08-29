import torch
import time
import psutil
import typing

def run_benchmark() -> dict[str, float]:
    # We use a larger matrix for GPUs to properly stress them.
    # If OOM occurs, we fallback to the smaller size.
    TARGET_MATRIX_SIZE = 4096 
    FALLBACK_MATRIX_SIZE = 2048
    WARMUP_ITERATIONS = 5
    MEASURE_ITERATIONS = 10
    
    #Detect Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ Starting compute benchmark on: {device}...")
    
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        print(f"   Hardware: {gpu_name}")
    
    torch.set_num_threads(1)
    
    #try for a larger matrix with the presence of gpu.
    matrix_size = TARGET_MATRIX_SIZE
    try:
        a = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
        b = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
    except RuntimeError:
        print(f"⚠️  Not enough memory for {matrix_size}x{matrix_size}, falling back to {FALLBACK_MATRIX_SIZE}...")
        matrix_size = FALLBACK_MATRIX_SIZE
        a = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
        b = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
    
    assert a.shape == (matrix_size, matrix_size)
    assert b.shape == (matrix_size, matrix_size)
    assert a.dtype == torch.float32
    assert b.dtype == torch.float32
    
    #Warmup
    for _ in range(WARMUP_ITERATIONS):
        result = torch.matmul(a, b)
    
    #Sync CUDA before starting timer
    if device.type == 'cuda':
        torch.cuda.synchronize()

    start_time = time.perf_counter()
    for _ in range(MEASURE_ITERATIONS):
        result = torch.matmul(a, b)
    
    #Sync CUDA before stopping timer
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.perf_counter()
    
    #Math to calculate flops
    total_time = end_time - start_time
    avg_time_per_op = total_time / MEASURE_ITERATIONS
    flops_per_op = 2 * (matrix_size ** 3)
    flops_per_second = flops_per_op / avg_time_per_op
    tflops = flops_per_second / 1e12
    
    assert total_time > 0, "❌ Total benchmark time must be positive"
    assert tflops > 0, "❌ TFLOPS measurement must be positive"
    
    print("✅ Starting memory benchmark...")
    
    # 5. Memory Detection (VRAM vs RAM)
    if device.type == 'cuda':
        # Get Total VRAM (in GB)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"   Detected VRAM: {mem_gb:.2f} GB")
    else:
        # Get Available System RAM
        mem_info = psutil.virtual_memory()
        mem_gb = mem_info.available / (1024 ** 3)
        print(f"   Detected System RAM: {mem_gb:.2f} GB")
    
    assert mem_gb > 0, "❌ Available memory must be positive"
    assert isinstance(mem_gb, float), "❌ Available memory must be a float"
    
    benchmark_results = {
        "flops": tflops,
        "memory_gb": mem_gb,
        #device capability:
        "device_type": "cuda" if device.type == "cuda" else "cpu"
    }
    
    assert isinstance(benchmark_results, dict)
    assert "flops" in benchmark_results
    assert "memory_gb" in benchmark_results
    assert isinstance(benchmark_results["flops"], float)
    assert isinstance(benchmark_results["memory_gb"], float)
    
    return benchmark_results

if __name__ == "__main__":
    print("✅ Running local device benchmark...")
    
    benchmark_results = run_benchmark()
    
    assert isinstance(benchmark_results, dict)
    assert "flops" in benchmark_results
    assert "memory_gb" in benchmark_results
    assert benchmark_results["flops"] > 0, f"❌ Invalid FLOPS value: {benchmark_results['flops']}"
    assert benchmark_results["memory_gb"] > 0, f"❌ Invalid memory value: {benchmark_results['memory_gb']}"
    assert isinstance(benchmark_results["flops"], float)
    assert isinstance(benchmark_results["memory_gb"], float)
    
    print("✅ Benchmark Complete")
    print("--------------------")
    print(f"✅ Compute: {benchmark_results['flops']:.4f} TFLOPS")
    print(f"✅ Memory:  {benchmark_results['memory_gb']:.2f} GB (VRAM/RAM)")
    print("--------------------")
    print("✅ This device is ready for distributed training.")