#!/usr/bin/env python3
"""
Qwen3 Thinking Mode Control Examples
Demonstrates how to enable/disable thinking mode for faster responses
"""

import requests
import json
import time

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:32b"

def test_with_thinking(prompt):
    """Test with thinking mode enabled (default)"""
    print("\n=== WITH THINKING MODE ===")
    start_time = time.time()
    
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1}
        # Note: thinking is enabled by default when think parameter is not specified
    }
    
    response = requests.post(url, json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        content = result['message']['content']
        print(f"Response ({elapsed:.1f}s):\n{content}\n")
    else:
        print(f"Error: {response.status_code}")

def test_without_thinking(prompt):
    """Test with thinking mode disabled for faster responses"""
    print("\n=== WITHOUT THINKING MODE ===")
    start_time = time.time()
    
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.1},
        "think": False  # Explicitly disable thinking mode
    }
    
    response = requests.post(url, json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        content = result['message']['content']
        print(f"Response ({elapsed:.1f}s):\n{content}\n")
    else:
        print(f"Error: {response.status_code}")

def test_openai_compatible_without_thinking(prompt):
    """Test OpenAI-compatible endpoint with thinking disabled"""
    print("\n=== OPENAI-COMPATIBLE API (NO THINKING) ===")
    start_time = time.time()
    
    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer ollama"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "think": False  # Disable thinking in OpenAI-compatible mode
    }
    
    response = requests.post(url, headers=headers, json=payload)
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        content = result['choices'][0]['message']['content']
        print(f"Response ({elapsed:.1f}s):\n{content}\n")
    else:
        print(f"Error: {response.status_code}")

def compare_thinking_modes():
    """Compare response times and outputs with and without thinking"""
    test_prompts = [
        "What is 25 + 37?",
        "Write a haiku about coding",
        "Explain photosynthesis in one sentence"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print('='*60)
        
        # Test without thinking (faster)
        test_without_thinking(prompt)
        
        # Test with thinking (more detailed but slower)
        test_with_thinking(prompt)

if __name__ == "__main__":
    print(f"Testing Qwen3 Thinking Mode Control")
    print(f"Model: {MODEL_NAME}")
    print(f"API URL: {OLLAMA_BASE_URL}")
    
    # Test simple math problem
    simple_prompt = "What is 10 + 15?"
    
    print("\n" + "="*60)
    print("COMPARING THINKING MODES")
    print("="*60)
    
    # Without thinking - fast response
    test_without_thinking(simple_prompt)
    
    # With thinking - shows reasoning process
    test_with_thinking(simple_prompt)
    
    # OpenAI-compatible without thinking
    test_openai_compatible_without_thinking(simple_prompt)
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print("\nKey Differences:")
    print("- WITHOUT thinking: Faster responses, direct answers")
    print("- WITH thinking: Shows reasoning process, slower but more detailed")
    print("\nUse 'think': False for:")
    print("- Simple queries")
    print("- Real-time applications")
    print("- When you don't need to see the reasoning")
    print("\nUse default (thinking enabled) for:")
    print("- Complex problems")
    print("- When you want to verify reasoning")
    print("- Educational purposes")