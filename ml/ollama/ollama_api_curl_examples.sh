#!/bin/bash
# Ollama API Examples using curl

echo "=== Ollama API Examples ==="
echo "API is running at: http://localhost:11434"
echo ""

echo "1. Native Ollama API - Simple Generation:"
echo 'curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen:14b",
    "prompt": "What is 2+2?",
    "stream": false
  }''
echo ""

echo "2. OpenAI-Compatible Chat Completions:"
echo 'curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ollama" \
  -d '{
    "model": "qwen:14b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello!"}
    ]
  }''
echo ""

echo "3. List Available Models:"
echo 'curl http://localhost:11434/api/tags'
echo ""

echo "4. Check Model Info:"
echo 'curl -X POST http://localhost:11434/api/show \
  -d '{"name": "qwen:14b"}''
echo ""

echo "5. For use with external applications:"
echo "- Base URL: http://localhost:11434/v1"
echo "- API Key: ollama (or any string)"
echo "- Model: qwen:14b"
echo ""

echo "6. To make API accessible from other machines:"
echo "Edit /etc/systemd/system/ollama.service and add:"
echo "Environment=\"OLLAMA_HOST=0.0.0.0\""
echo "Then restart: sudo systemctl restart ollama"