#!/usr/bin/env python3
"""
Real-time monitoring of Ollama during parallel requests
"""

import subprocess
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor

# Flag to control monitoring
monitoring = True

def monitor_resources():
    """Monitor and print resources in real-time"""
    print("\nTime     | GPU Mem  | GPU%  | Ollama CPU% | Status")
    print("-" * 60)
    
    while monitoring:
        try:
            # Get GPU stats
            gpu_result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True)
            
            gpu_mem, gpu_util = gpu_result.stdout.strip().split(', ')
            
            # Get Ollama CPU
            pid_result = subprocess.run(['pidof', 'ollama'], capture_output=True, text=True)
            if pid_result.returncode == 0:
                pid = pid_result.stdout.strip().split()[0]
                cpu_result = subprocess.run(
                    f"ps -p {pid} -o %cpu --no-headers",
                    shell=True,
                    capture_output=True, 
                    text=True
                )
                ollama_cpu = float(cpu_result.stdout.strip()) if cpu_result.stdout.strip() else 0
            else:
                ollama_cpu = 0
            
            # Print current status
            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp} | {gpu_mem:>7s}M | {gpu_util:>4s}% | {ollama_cpu:>10.1f}% | Running", end='\r')
            
        except Exception as e:
            print(f"Monitor error: {e}", end='\r')
        
        time.sleep(0.5)

def make_request(req_id):
    """Make a simple request"""
    start = time.time()
    response = requests.post(
        'http://localhost:11434/api/generate',
        json={
            'model': 'qwen3:32b',
            'prompt': f'Write a story about the number {req_id}. Make it exactly 100 words.',
            'stream': False,
            'options': {'temperature': 0}
        },
        timeout=120
    )
    duration = time.time() - start
    status = "OK" if response.status_code == 200 else f"Error {response.status_code}"
    print(f"\nRequest {req_id} completed: {duration:.1f}s - {status}")
    return req_id, duration

def main():
    global monitoring
    
    print("Real-time Ollama Resource Monitor")
    print("Watching for CPU offloading during parallel requests...")
    
    # Start monitoring in background
    monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
    monitor_thread.start()
    
    # Wait a moment
    time.sleep(2)
    
    print("\n\n=== TEST 1: Single Request ===")
    time.sleep(1)
    make_request(1)
    
    time.sleep(3)
    
    print("\n\n=== TEST 2: Two Parallel Requests ===")
    print("(Watch for CPU spike indicating offloading)")
    time.sleep(1)
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(make_request, i) for i in [2, 3]]
        for future in futures:
            future.result()
    
    time.sleep(3)
    
    print("\n\n=== TEST 3: Four Parallel Requests ===")
    print("(Should definitely show CPU offloading if memory is exceeded)")
    time.sleep(1)
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(make_request, i) for i in [4, 5, 6, 7]]
        for future in futures:
            future.result()
    
    # Stop monitoring
    monitoring = False
    time.sleep(1)
    
    print("\n\nTest complete. If you saw Ollama CPU% spike to 1000%+, that indicates CPU offloading.")

if __name__ == "__main__":
    main()