#!/usr/bin/env python3
"""
Example script showing how to use Ollama's API with both native and OpenAI-compatible endpoints
"""

import requests
import json

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "qwen3:32b"

def test_native_ollama_api():
    """Test the native Ollama API endpoint"""
    print("=== Testing Native Ollama API ===")
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": "Write a haiku about programming",
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['response']}")
        print(f"Total duration: {result['total_duration'] / 1e9:.2f} seconds")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    print()

def test_openai_compatible_api():
    """Test the OpenAI-compatible chat completions endpoint"""
    print("=== Testing OpenAI-Compatible API ===")
    
    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer ollama"  # Any value works, just needs to be present
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain what an API is in simple terms"}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {result['choices'][0]['message']['content']}")
        print(f"Tokens used: {result['usage']['total_tokens']}")
        print(f"Model: {result['model']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
    print()

def test_streaming_api():
    """Test the streaming API endpoint"""
    print("=== Testing Streaming API ===")
    
    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer ollama"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "Count from 1 to 5 slowly"}
        ],
        "stream": True
    }
    
    response = requests.post(url, headers=headers, json=payload, stream=True)
    if response.status_code == 200:
        print("Streaming response: ", end="", flush=True)
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk['choices'][0]['delta'].get('content', '')
                        print(content, end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        print("\n")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def use_with_openai_library():
    """Example of using the OpenAI Python library with Ollama"""
    print("=== Using OpenAI Python Library ===")
    print("To use with OpenAI's Python library:")
    print("""
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Any value works
)

response = client.chat.completions.create(
    model="qwen:14b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
""")

if __name__ == "__main__":
    print(f"Testing Ollama API with model: {MODEL_NAME}")
    print(f"Base URL: {OLLAMA_BASE_URL}\n")
    
    # Test native API
    test_native_ollama_api()
    
    # Test OpenAI-compatible API
    test_openai_compatible_api()
    
    # Test streaming
    test_streaming_api()
    
    # Show OpenAI library usage
    use_with_openai_library()