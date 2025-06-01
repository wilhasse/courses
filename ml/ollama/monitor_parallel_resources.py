#!/usr/bin/env python3
"""
Monitor CPU and GPU resources during Ollama parallel processing
"""

import requests
import time
import threading
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

# Global storage for metrics
metrics = {
    'cpu': [],
    'gpu': [],
    'requests': []
}
monitoring = True
metrics_lock = threading.Lock()

def get_cpu_usage():
    """Get current CPU usage percentage"""
    try:
        result = subprocess.run(['top', '-bn1'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'Cpu(s)' in line or '%Cpu' in line:
                # Extract idle percentage and calculate usage
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'id' in part and i > 0:
                        idle = float(parts[i-1].replace(',', ''))
                        return 100.0 - idle
        return 0
    except:
        return 0

def get_gpu_metrics():
    """Get GPU memory and utilization"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=memory.used,memory.total,utilization.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            values = result.stdout.strip().split(', ')
            return {
                'memory_used': int(values[0]),
                'memory_total': int(values[1]),
                'gpu_util': int(values[2]),
                'power': float(values[3])
            }
    except:
        pass
    return None

def get_ollama_process_cpu():
    """Get Ollama process CPU usage"""
    try:
        result = subprocess.run(['pidof', 'ollama'], capture_output=True, text=True)
        if result.returncode == 0:
            pid = result.stdout.strip().split()[0]
            ps_result = subprocess.run(['ps', '-p', pid, '-o', '%cpu', '--no-headers'], 
                                     capture_output=True, text=True)
            if ps_result.returncode == 0:
                return float(ps_result.stdout.strip())
    except:
        pass
    return 0

def monitor_resources():
    """Background thread to monitor resources"""
    global monitoring
    
    while monitoring:
        timestamp = time.time()
        
        # Get metrics
        cpu_total = get_cpu_usage()
        ollama_cpu = get_ollama_process_cpu()
        gpu = get_gpu_metrics()
        
        with metrics_lock:
            metrics['cpu'].append({
                'timestamp': timestamp,
                'total_cpu': cpu_total,
                'ollama_cpu': ollama_cpu
            })
            
            if gpu:
                metrics['gpu'].append({
                    'timestamp': timestamp,
                    'memory_used': gpu['memory_used'],
                    'memory_total': gpu['memory_total'],
                    'gpu_util': gpu['gpu_util'],
                    'power': gpu['power']
                })
        
        time.sleep(0.5)  # Sample every 500ms

def make_request(request_id, prompt_tokens=100):
    """Make a request and track timing"""
    start_time = time.time()
    
    # Create a prompt that generates predictable output length
    prompt = f"Count from {request_id*10} to {request_id*10 + 20}. Be detailed."
    
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'qwen3:32b',
                'prompt': prompt,
                'stream': False,
                'options': {
                    'temperature': 0,
                    'num_predict': prompt_tokens
                }
            },
            timeout=120
        )
        
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            with metrics_lock:
                metrics['requests'].append({
                    'id': request_id,
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'tokens': data.get('eval_count', 0),
                    'eval_duration': data.get('eval_duration', 0) / 1e9,
                    'status': 'success'
                })
            return request_id, end_time - start_time, 'success'
        else:
            with metrics_lock:
                metrics['requests'].append({
                    'id': request_id,
                    'start': start_time,
                    'end': end_time,
                    'duration': end_time - start_time,
                    'status': f'error: {response.status_code}'
                })
            return request_id, end_time - start_time, f'error: {response.status_code}'
            
    except Exception as e:
        end_time = time.time()
        with metrics_lock:
            metrics['requests'].append({
                'id': request_id,
                'start': start_time,
                'end': end_time,
                'duration': end_time - start_time,
                'status': f'exception: {str(e)}'
            })
        return request_id, end_time - start_time, f'exception: {str(e)}'

def run_test(num_parallel, description):
    """Run a test with monitoring"""
    global metrics
    
    # Clear previous metrics
    with metrics_lock:
        metrics = {'cpu': [], 'gpu': [], 'requests': []}
    
    print(f"\n{'='*70}")
    print(f"Test: {description}")
    print(f"Parallel requests: {num_parallel}")
    print(f"{'='*70}")
    
    # Start monitoring
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    # Wait for baseline metrics
    time.sleep(2)
    
    test_start = time.time()
    
    # Run parallel requests
    with ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = [executor.submit(make_request, i) for i in range(num_parallel)]
        for future in as_completed(futures):
            req_id, duration, status = future.result()
            print(f"Request {req_id}: {duration:.2f}s - {status}")
    
    test_end = time.time()
    test_duration = test_end - test_start
    
    # Continue monitoring for a bit
    time.sleep(2)
    
    # Analyze metrics
    with metrics_lock:
        # CPU metrics
        cpu_during_test = [m for m in metrics['cpu'] if test_start <= m['timestamp'] <= test_end]
        if cpu_during_test:
            avg_total_cpu = sum(m['total_cpu'] for m in cpu_during_test) / len(cpu_during_test)
            max_total_cpu = max(m['total_cpu'] for m in cpu_during_test)
            avg_ollama_cpu = sum(m['ollama_cpu'] for m in cpu_during_test) / len(cpu_during_test)
            max_ollama_cpu = max(m['ollama_cpu'] for m in cpu_during_test)
        else:
            avg_total_cpu = max_total_cpu = avg_ollama_cpu = max_ollama_cpu = 0
        
        # GPU metrics
        gpu_before = [m for m in metrics['gpu'] if m['timestamp'] < test_start]
        gpu_during = [m for m in metrics['gpu'] if test_start <= m['timestamp'] <= test_end]
        
        if gpu_before and gpu_during:
            baseline_mem = sum(m['memory_used'] for m in gpu_before[-3:]) / min(3, len(gpu_before))
            peak_mem = max(m['memory_used'] for m in gpu_during)
            avg_mem = sum(m['memory_used'] for m in gpu_during) / len(gpu_during)
            avg_gpu_util = sum(m['gpu_util'] for m in gpu_during) / len(gpu_during)
            max_gpu_util = max(m['gpu_util'] for m in gpu_during)
        else:
            baseline_mem = peak_mem = avg_mem = avg_gpu_util = max_gpu_util = 0
    
    print(f"\nPerformance Summary:")
    print(f"- Test duration: {test_duration:.2f}s")
    print(f"- Requests completed: {len([r for r in metrics['requests'] if r.get('status') == 'success'])}")
    
    print(f"\nCPU Usage:")
    print(f"- Average total CPU: {avg_total_cpu:.1f}%")
    print(f"- Peak total CPU: {max_total_cpu:.1f}%")
    print(f"- Average Ollama CPU: {avg_ollama_cpu:.1f}%")
    print(f"- Peak Ollama CPU: {max_ollama_cpu:.1f}%")
    
    if max_ollama_cpu > 200:
        print(f"  ⚠️  WARNING: High CPU usage detected! Possible CPU offloading.")
    
    print(f"\nGPU Memory:")
    print(f"- Baseline: {baseline_mem:.0f} MB")
    print(f"- Peak during test: {peak_mem:.0f} MB")
    print(f"- Memory increase: {peak_mem - baseline_mem:.0f} MB")
    print(f"- GPU utilization: {avg_gpu_util:.0f}% avg, {max_gpu_util:.0f}% peak")
    
    return test_duration, max_ollama_cpu, peak_mem

def main():
    global monitoring
    
    print("Ollama Parallel Processing Resource Monitor")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check initial state
    print("\nInitial GPU State:")
    gpu = get_gpu_metrics()
    if gpu:
        print(f"- Memory: {gpu['memory_used']} / {gpu['memory_total']} MB")
        print(f"- GPU Utilization: {gpu['gpu_util']}%")
    
    # Warm up
    print("\nWarming up model...")
    make_request(99, prompt_tokens=10)
    time.sleep(3)
    
    # Run tests
    results = []
    
    # Test 1: Single request
    duration1, cpu1, mem1 = run_test(1, "Single Request Baseline")
    results.append((1, duration1, cpu1, mem1))
    time.sleep(5)
    
    # Test 2: Two parallel
    duration2, cpu2, mem2 = run_test(2, "Two Parallel Requests")
    results.append((2, duration2, cpu2, mem2))
    time.sleep(5)
    
    # Test 3: Three parallel (should show queueing)
    duration3, cpu3, mem3 = run_test(3, "Three Parallel Requests")
    results.append((3, duration3, cpu3, mem3))
    
    # Stop monitoring
    monitoring = False
    
    # Final analysis
    print("\n" + "="*70)
    print("FINAL ANALYSIS")
    print("="*70)
    
    for n, duration, cpu, mem in results:
        print(f"\n{n} Parallel Request(s):")
        print(f"- Duration: {duration:.2f}s")
        print(f"- Peak Ollama CPU: {cpu:.0f}%")
        print(f"- Peak GPU Memory: {mem:.0f} MB")
        
        if cpu > 500:
            print("  🚨 SEVERE CPU OFFLOADING DETECTED!")
        elif cpu > 200:
            print("  ⚠️  Some CPU offloading detected")
        else:
            print("  ✓ Running on GPU")
    
    print("\nConclusions:")
    if any(r[2] > 500 for r in results):
        print("❌ Model is being offloaded to CPU during parallel processing!")
        print("   This explains the 1000% CPU usage you observed.")
        print("   Even with NUM_PARALLEL=2, memory requirements exceed VRAM.")
    else:
        print("✓ Model appears to be running on GPU")

if __name__ == "__main__":
    main()