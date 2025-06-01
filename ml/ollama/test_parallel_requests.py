#!/usr/bin/env python3
"""
Test Ollama's parallel request handling capabilities
"""

import requests
import concurrent.futures
import time
import json
from datetime import datetime

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:32b"

def make_request(request_id):
    """Make a single request to Ollama API"""
    start_time = time.time()
    
    payload = {
        "model": MODEL_NAME,
        "prompt": f"Say 'Hello from request {request_id}' and nothing else",
        "stream": False,
        "options": {
            "temperature": 0.1,
            "seed": request_id  # Use different seed for each request
        }
    }
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            duration = time.time() - start_time
            return {
                "request_id": request_id,
                "status": "success",
                "response": result.get('response', '').strip(),
                "duration": duration,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "request_id": request_id,
                "status": "error",
                "error": f"HTTP {response.status_code}",
                "duration": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
    except Exception as e:
        return {
            "request_id": request_id,
            "status": "exception",
            "error": str(e),
            "duration": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }

def test_parallel_requests(num_requests):
    """Test with specified number of parallel requests"""
    print(f"\n=== Testing {num_requests} Parallel Requests ===")
    print(f"Model: {MODEL_NAME}")
    print(f"Start time: {datetime.now().isoformat()}\n")
    
    overall_start = time.time()
    
    # Submit all requests in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = {executor.submit(make_request, i): i for i in range(1, num_requests + 1)}
        results = []
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            
            # Print result immediately
            if result['status'] == 'success':
                print(f"✓ Request {result['request_id']:2d} completed in {result['duration']:6.2f}s: {result['response']}")
            else:
                print(f"✗ Request {result['request_id']:2d} failed in {result['duration']:6.2f}s: {result['error']}")
    
    overall_duration = time.time() - overall_start
    
    # Summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful
    avg_duration = sum(r['duration'] for r in results) / len(results)
    
    print(f"\nSummary:")
    print(f"- Total requests: {num_requests}")
    print(f"- Successful: {successful}")
    print(f"- Failed: {failed}")
    print(f"- Overall duration: {overall_duration:.2f}s")
    print(f"- Average request duration: {avg_duration:.2f}s")
    print(f"- Requests per second: {num_requests / overall_duration:.2f}")
    
    return results

def check_server_info():
    """Check Ollama server information"""
    print("=== Ollama Server Information ===")
    
    # Check version
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/version")
        if response.status_code == 200:
            print(f"Version: {response.json().get('version', 'Unknown')}")
    except:
        pass
    
    # Check loaded models
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/ps")
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"Loaded models: {len(models)}")
            for model in models:
                print(f"  - {model['name']} (expires: {model['expires_at']})")
    except:
        pass

if __name__ == "__main__":
    # Check server info
    check_server_info()
    
    # Test with increasing number of parallel requests
    test_scenarios = [1, 2, 4, 8, 16]
    
    print("\nStarting parallel request tests...")
    print("Note: Ollama's default OLLAMA_NUM_PARALLEL is 4 (or 1 with limited memory)")
    print("      Default OLLAMA_MAX_QUEUE is 512")
    
    for num_requests in test_scenarios:
        results = test_parallel_requests(num_requests)
        
        # Small delay between tests
        if num_requests < test_scenarios[-1]:
            time.sleep(2)
    
    print("\n=== Test Complete ===")
    print("If requests are being processed sequentially, you may need to:")
    print("1. Set OLLAMA_NUM_PARALLEL environment variable")
    print("2. Ensure sufficient GPU memory for parallel processing")
    print("3. Check Ollama logs: journalctl -u ollama -f")