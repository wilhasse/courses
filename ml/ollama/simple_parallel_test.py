#!/usr/bin/env python3
"""
Simple parallel performance test for Ollama
"""

import requests
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def test_request(request_id):
    """Make a simple test request"""
    start = time.time()
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen3:32b",
            "prompt": f"Say only the number {request_id}",
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 10  # Limit response length
            }
        },
        timeout=60
    )
    
    duration = time.time() - start
    
    if response.status_code == 200:
        data = response.json()
        tokens = data.get('eval_count', 0)
        eval_duration = data.get('eval_duration', 0) / 1e9
        tokens_per_sec = tokens / eval_duration if eval_duration > 0 else 0
        return request_id, duration, tokens_per_sec, "success"
    else:
        return request_id, duration, 0, f"error: {response.status_code}"

print("Testing Ollama GPU Parallel Processing")
print("=" * 60)

# Test 1: Single request
print("\nTest 1: Single Request")
start = time.time()
result = test_request(1)
single_time = time.time() - start
print(f"Request {result[0]}: {result[1]:.2f}s, {result[2]:.1f} tokens/s")

time.sleep(2)

# Test 2: Two parallel requests
print("\nTest 2: Two Parallel Requests")
start = time.time()
with ThreadPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(test_request, i) for i in [1, 2]]
    for future in as_completed(futures):
        result = future.result()
        print(f"Request {result[0]}: {result[1]:.2f}s, {result[2]:.1f} tokens/s")
parallel_time = time.time() - start

print("\n" + "=" * 60)
print("RESULTS:")
print(f"- Single request time: {single_time:.2f}s")
print(f"- Two parallel requests time: {parallel_time:.2f}s")
print(f"- Parallel time ratio: {parallel_time/single_time:.2f}x (ideal=1.0x, sequential=2.0x)")
print(f"- Parallel efficiency: {(single_time * 2) / parallel_time / 2 * 100:.1f}%")

print("\nGPU PARALLELISM EXPLANATION:")
print("""
With sufficient GPU memory, parallel processing works by:

1. **Batching**: GPU processes multiple sequences simultaneously
   - KV-cache (attention) is computed for each sequence
   - Matrix operations can be batched efficiently

2. **Expected Performance**:
   - 2 parallel requests typically take 1.2-1.5x single request time
   - Not 2x because GPU can batch operations
   - Not 1x due to memory bandwidth competition

3. **Memory Requirements**:
   - Each parallel request needs its own KV-cache
   - Context memory multiplies by NUM_PARALLEL
   - Qwen3 32B needs ~10GB per parallel context
""")