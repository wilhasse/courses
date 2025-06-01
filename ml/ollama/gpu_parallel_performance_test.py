#!/usr/bin/env python3
"""
Test GPU parallel processing performance with detailed timing
"""

import requests
import concurrent.futures
import time
import threading
import json

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:32b"

# Thread-safe list to store timing data
timing_data = []
timing_lock = threading.Lock()

def make_request(request_id, prompt_length="short"):
    """Make a request and track detailed timing"""
    if prompt_length == "short":
        prompt = f"What is {request_id} + {request_id}?"
    else:
        prompt = f"Write a detailed explanation of why {request_id} + {request_id} = {2*request_id}. Include the mathematical principles."
    
    start_time = time.time()
    queue_time = start_time
    
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=120
        )
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            eval_count = result.get('eval_count', 0)
            prompt_eval_count = result.get('prompt_eval_count', 0)
            eval_duration = result.get('eval_duration', 0) / 1e9  # Convert to seconds
            prompt_eval_duration = result.get('prompt_eval_duration', 0) / 1e9
            
            # Calculate tokens per second
            eval_rate = eval_count / eval_duration if eval_duration > 0 else 0
            
            with timing_lock:
                timing_data.append({
                    'request_id': request_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'total_duration': total_duration,
                    'eval_duration': eval_duration,
                    'prompt_eval_duration': prompt_eval_duration,
                    'eval_count': eval_count,
                    'prompt_eval_count': prompt_eval_count,
                    'eval_rate': eval_rate,
                    'status': 'success'
                })
            
            return request_id, total_duration, eval_rate, eval_count
        else:
            with timing_lock:
                timing_data.append({
                    'request_id': request_id,
                    'start_time': start_time,
                    'end_time': end_time,
                    'total_duration': total_duration,
                    'status': 'error',
                    'error_code': response.status_code
                })
            return request_id, total_duration, 0, 0
            
    except Exception as e:
        end_time = time.time()
        with timing_lock:
            timing_data.append({
                'request_id': request_id,
                'start_time': start_time,
                'end_time': end_time,
                'total_duration': end_time - start_time,
                'status': 'exception',
                'error': str(e)
            })
        return request_id, end_time - start_time, 0, 0

def run_test(num_parallel, prompt_length="short"):
    """Run a test with specified parallelism"""
    global timing_data
    timing_data = []
    
    print(f"\n{'='*70}")
    print(f"Testing {num_parallel} {'parallel' if num_parallel > 1 else 'sequential'} request(s) - {prompt_length} prompt")
    print(f"{'='*70}")
    
    overall_start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel) as executor:
        futures = [executor.submit(make_request, i, prompt_length) for i in range(1, num_parallel + 1)]
        
        for future in concurrent.futures.as_completed(futures):
            request_id, duration, eval_rate, token_count = future.result()
            print(f"Request {request_id}: {duration:.2f}s total, {eval_rate:.1f} tokens/s, {token_count} tokens")
    
    overall_duration = time.time() - overall_start
    
    # Analyze timing overlap
    if len(timing_data) > 1:
        timing_data.sort(key=lambda x: x['start_time'])
        
        # Check for parallel execution
        max_concurrent = 1
        for i in range(len(timing_data)):
            concurrent_count = 1
            for j in range(len(timing_data)):
                if i != j:
                    # Check if request j was running during request i
                    if (timing_data[j]['start_time'] < timing_data[i]['end_time'] and 
                        timing_data[j]['end_time'] > timing_data[i]['start_time']):
                        concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
        
        print(f"\nParallel Execution Analysis:")
        print(f"- Maximum concurrent requests: {max_concurrent}")
        print(f"- Overall duration: {overall_duration:.2f}s")
        
        # Calculate average stats
        successful_requests = [d for d in timing_data if d.get('status') == 'success']
        if successful_requests:
            avg_duration = sum(d['total_duration'] for d in successful_requests) / len(successful_requests)
            avg_eval_rate = sum(d.get('eval_rate', 0) for d in successful_requests) / len(successful_requests)
            total_tokens = sum(d.get('eval_count', 0) for d in successful_requests)
            
            print(f"- Average request duration: {avg_duration:.2f}s")
            print(f"- Average tokens/s per request: {avg_eval_rate:.1f}")
            print(f"- Total tokens generated: {total_tokens}")
            print(f"- Overall tokens/s: {total_tokens / overall_duration:.1f}")
            
            # Performance comparison
            if num_parallel > 1:
                speedup = (avg_duration * num_parallel) / overall_duration
                efficiency = speedup / num_parallel * 100
                print(f"- Speedup: {speedup:.2f}x")
                print(f"- Parallel efficiency: {efficiency:.1f}%")
    
    return overall_duration

def main():
    print("GPU Parallel Processing Performance Test")
    print("Model: qwen3:32b on RTX 4090")
    print("\nNote: OLLAMA_NUM_PARALLEL=2 (configured)")
    
    # Warm up the model
    print("\nWarming up model...")
    make_request(0, "short")
    time.sleep(2)
    
    # Test 1: Sequential vs Parallel with short prompts
    print("\n" + "="*70)
    print("TEST 1: Short prompts (math questions)")
    print("="*70)
    
    seq_time = run_test(1, "short")
    time.sleep(3)
    par_time = run_test(2, "short")
    
    print(f"\n**Summary for short prompts:**")
    print(f"- Sequential (1 request): {seq_time:.2f}s")
    print(f"- Parallel (2 requests): {par_time:.2f}s")
    print(f"- Time ratio: {par_time/seq_time:.2f}x (ideal would be 1.0x)")
    
    time.sleep(3)
    
    # Test 2: Sequential vs Parallel with longer prompts
    print("\n" + "="*70)
    print("TEST 2: Longer prompts (detailed explanations)")
    print("="*70)
    
    seq_time_long = run_test(1, "long")
    time.sleep(3)
    par_time_long = run_test(2, "long")
    
    print(f"\n**Summary for longer prompts:**")
    print(f"- Sequential (1 request): {seq_time_long:.2f}s")
    print(f"- Parallel (2 requests): {par_time_long:.2f}s")
    print(f"- Time ratio: {par_time_long/seq_time_long:.2f}x (ideal would be 1.0x)")
    
    # GPU utilization info
    print("\n" + "="*70)
    print("GPU PARALLEL PROCESSING EXPLANATION:")
    print("="*70)
    print("""
1. **Batch Processing**: GPUs can process multiple sequences in a batch
   - More efficient than sequential processing
   - But not perfectly parallel due to memory bandwidth limits

2. **Memory Bandwidth**: The main bottleneck for LLMs
   - Model weights need to be loaded from VRAM for each token
   - Parallel requests compete for memory bandwidth

3. **Compute vs Memory Bound**:
   - LLMs are typically memory-bandwidth bound, not compute bound
   - This limits parallel speedup

4. **Expected Performance**:
   - 2 parallel requests typically take 1.2-1.5x the time of 1 request
   - Not 2x because GPU can batch some operations
   - Not 1x because memory bandwidth is shared
""")

if __name__ == "__main__":
    main()