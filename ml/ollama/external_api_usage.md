# Ollama API External Usage Guide

## Current Setup
- **API URL**: `http://localhost:11434` (local only)
- **Model**: `qwen:14b`
- **Endpoints**:
  - Native: `/api/generate`
  - OpenAI-compatible: `/v1/chat/completions`

## Using from External Applications

### 1. With OpenAI Python Library
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://10.1.1.198:11434/v1",  # After enabling external access
    api_key="ollama"  # Any string works
)

response = client.chat.completions.create(
    model="qwen:14b",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### 2. With LangChain
```python
from langchain_community.llms import Ollama

llm = Ollama(
    base_url="http://10.1.1.198:11434",
    model="qwen:14b"
)

response = llm.invoke("What is the meaning of life?")
```

### 3. With curl from Remote Machine
```bash
curl -X POST http://10.1.1.198:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ollama" \
  -d '{
    "model": "qwen:14b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 4. With JavaScript/Node.js
```javascript
const response = await fetch('http://10.1.1.198:11434/v1/chat/completions', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ollama'
  },
  body: JSON.stringify({
    model: 'qwen:14b',
    messages: [{ role: 'user', content: 'Hello!' }]
  })
});

const data = await response.json();
console.log(data.choices[0].message.content);
```

## API Compatibility
The `/v1/chat/completions` endpoint is compatible with:
- OpenAI API clients
- ChatGPT plugins
- Auto-GPT
- LangChain
- Llama Index
- And most OpenAI-compatible tools

## Rate Limiting
Ollama doesn't have built-in rate limiting, so implement your own if needed for production use.