#!/usr/bin/env python3
"""
Simple test of Ollama parallel request handling
"""

import requests
import concurrent.futures
import time

def make_request(i):
    start = time.time()
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                'model': 'qwen3:32b',
                'prompt': f'What is {i} + {i}?',
                'stream': False,
                'options': {'temperature': 0.1}
            },
            timeout=120
        )
        duration = time.time() - start
        if response.status_code == 200:
            result = response.json()
            answer = result.get('response', '').strip()
            # Extract just the answer part, remove thinking tags
            if '<think>' in answer:
                answer = answer.split('</think>')[-1].strip()
            return f"Request {i}: {duration:.2f}s - Answer: {answer[:50]}..."
        else:
            return f"Request {i}: {duration:.2f}s - Error: {response.status_code}"
    except Exception as e:
        duration = time.time() - start
        return f"Request {i}: {duration:.2f}s - Exception: {str(e)}"

print("Testing Ollama parallel processing with new configuration")
print("OLLAMA_NUM_PARALLEL=8, OLLAMA_MAX_QUEUE=1024")
print("="*60)

# Test different numbers of parallel requests
for num_requests in [1, 2, 4, 8]:
    print(f"\nTesting {num_requests} parallel requests:")
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(make_request, i) for i in range(1, num_requests + 1)]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())
    
    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    print(f"Requests per second: {num_requests / total_time:.2f}")
    
    # Give a small break between tests
    if num_requests < 8:
        time.sleep(2)