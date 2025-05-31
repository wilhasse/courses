#!/bin/bash
# Qwen3 API Examples - Controlling Thinking Mode

echo "=== Qwen3 Thinking Mode Control Examples ==="
echo ""

echo "1. DEFAULT (With Thinking) - Shows reasoning process:"
echo 'curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:32b",
    "messages": [{"role": "user", "content": "What is 10 + 15?"}],
    "stream": false
  }''
echo ""

echo "2. FAST MODE (Without Thinking) - Direct answer only:"
echo 'curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:32b",
    "messages": [{"role": "user", "content": "What is 10 + 15?"}],
    "stream": false,
    "think": false
  }''
echo ""

echo "3. OpenAI-Compatible API (Without Thinking):"
echo 'curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ollama" \
  -d '{
    "model": "qwen3:32b",
    "messages": [{"role": "user", "content": "What is 10 + 15?"}],
    "think": false
  }''
echo ""

echo "4. Streaming Response (Without Thinking):"
echo 'curl -X POST http://localhost:11434/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:32b",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true,
    "think": false
  }''
echo ""

echo "=== Usage Notes ==="
echo "- Add \"think\": false to disable thinking mode"
echo "- Thinking mode is enabled by default if not specified"
echo "- Disabling thinking significantly improves response time"
echo "- Use thinking mode for complex problems requiring step-by-step reasoning"
echo ""

echo "=== Python Example ==="
echo 'import requests

# Fast mode without thinking
response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "qwen3:32b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "think": False  # Key parameter to disable thinking
    }
)'