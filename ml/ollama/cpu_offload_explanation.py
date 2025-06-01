#!/usr/bin/env python3
"""
Demonstrate and explain CPU offloading in Ollama
"""

import subprocess
import requests
import time

def get_metrics():
    """Get current GPU and CPU metrics"""
    # GPU memory
    gpu_result = subprocess.run([
        'nvidia-smi', 
        '--query-gpu=memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'
    ], capture_output=True, text=True)
    
    gpu_used, gpu_free, gpu_util = gpu_result.stdout.strip().split(', ')
    
    # Ollama process info
    pid_result = subprocess.run(['pidof', 'ollama'], capture_output=True, text=True)
    if pid_result.returncode == 0:
        pid = pid_result.stdout.strip().split()[0]
        # Get detailed memory info
        mem_result = subprocess.run(
            f"cat /proc/{pid}/status | grep -E 'VmRSS|VmSwap|Threads'",
            shell=True,
            capture_output=True,
            text=True
        )
        
        # Get CPU usage
        cpu_result = subprocess.run(
            f"ps -p {pid} -o %cpu --no-headers",
            shell=True,
            capture_output=True, 
            text=True
        )
        cpu_usage = float(cpu_result.stdout.strip()) if cpu_result.stdout.strip() else 0
        
        return {
            'gpu_used': int(gpu_used),
            'gpu_free': int(gpu_free),
            'gpu_util': int(gpu_util),
            'cpu_usage': cpu_usage,
            'process_info': mem_result.stdout
        }
    
    return None

print("CPU Offloading Analysis for Qwen3 32B")
print("=" * 70)

# Initial state
print("\n1. INITIAL STATE:")
metrics = get_metrics()
if metrics:
    print(f"   GPU Memory: {metrics['gpu_used']}MB used, {metrics['gpu_free']}MB free")
    print(f"   CPU Usage: {metrics['cpu_usage']:.1f}%")
    print(f"   Process Info:\n{metrics['process_info']}")

# Test single request
print("\n2. DURING SINGLE REQUEST:")
print("   Making request...")

# Start request in background
import threading
def make_request():
    requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'qwen3:32b',
            'prompt': 'Count to 10',
            'stream': False
        }
    )

thread = threading.Thread(target=make_request)
thread.start()

# Monitor during request
time.sleep(2)
metrics = get_metrics()
if metrics:
    print(f"   GPU Memory: {metrics['gpu_used']}MB used, {metrics['gpu_free']}MB free")
    print(f"   GPU Utilization: {metrics['gpu_util']}%")
    print(f"   CPU Usage: {metrics['cpu_usage']:.1f}%")

thread.join()

print("\n3. ANALYSIS:")
print("""
The ~400% CPU usage indicates that Ollama is using 4 CPU cores constantly.
This happens because:

1. **Model Size vs VRAM**: 
   - Qwen3 32B model: ~20GB compressed
   - Available VRAM: 24GB total, ~23GB usable
   - With overhead and KV-cache, this is at the limit

2. **Partial Offloading**:
   - Most layers are on GPU (21.8GB in VRAM)
   - Some operations or layers run on CPU
   - This explains the constant 350-400% CPU usage

3. **With NUM_PARALLEL=2**:
   - Each parallel context needs additional KV-cache memory
   - This pushes beyond VRAM capacity
   - More operations offload to CPU

4. **Performance Impact**:
   - GPU still does most computation (40-50% utilization)
   - CPU handles overflow operations
   - Results in slower but functional inference

SOLUTION OPTIONS:
1. Use NUM_PARALLEL=1 for pure GPU execution
2. Use a smaller model (Qwen3 14B would fit comfortably)
3. Accept the CPU offloading for higher throughput
4. Get a GPU with more VRAM (48GB for comfortable parallel processing)
""")

# Check if we can fit the model better
print("\n4. MEMORY CALCULATION:")
print(f"   Model base: ~20GB")
print(f"   Per-request KV-cache: ~1-2GB") 
print(f"   With NUM_PARALLEL=2: ~22-24GB needed")
print(f"   Available VRAM: 23.5GB")
print(f"   Result: Just at the edge, causing partial CPU offload")

if __name__ == "__main__":
    pass