OLLAMA SETUP AND CONFIGURATION - COMPLETE GUIDE
===============================================

This document describes the complete setup process for Ollama with the Qwen 14B model
and enabling external API access.

SYSTEM INFORMATION
------------------
- GPU: NVIDIA GeForce RTX 4090 (24GB VRAM)
- OS: Linux 6.11.0-26-generic
- Ollama Version: 0.9.0
- Model: qwen:14b (8.2GB)
- Date: May 31, 2025

STEP 1: VERIFY GPU COMPATIBILITY
--------------------------------
Command: nvidia-smi
Result: RTX 4090 with 24GB VRAM detected - excellent for running large language models

STEP 2: CHECK OLLAMA INSTALLATION
---------------------------------
Command: which ollama
Result: /usr/local/bin/ollama (already installed)

Command: ollama --version
Result: ollama version is 0.9.0

STEP 3: DOWNLOAD QWEN MODEL
---------------------------
Note: There's no specific 13B model, so we used the closest available: 14B

Command: ollama pull qwen:14b
Result: Successfully downloaded 8.2GB model

Verification:
Command: ollama list
Output: qwen:14b    80362ced6553    8.2 GB    13 seconds ago

STEP 4: TEST THE MODEL
----------------------
Command: echo "What is the capital of France?" | ollama run qwen:14b
Result: "The capital of France is Paris."

Command: echo "Write a Python function that calculates the fibonacci sequence up to n terms" | ollama run qwen:14b
Result: Successfully generated working Python code

GPU Usage Check:
Command: nvidia-smi
Result: ~15GB VRAM used by Ollama process

STEP 5: VERIFY API SERVICE
--------------------------
Command: ps aux | grep ollama
Result: ollama serve process running on PID 1362

Command: curl -s http://localhost:11434/api/version
Result: {"version":"0.9.0"}

STEP 6: TEST NATIVE OLLAMA API
-------------------------------
Command: 
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen:14b",
    "prompt": "What is the capital of Japan?",
    "stream": false
  }'

Result: Successfully returned "The capital of Japan is Tokyo."

STEP 7: TEST OPENAI-COMPATIBLE API
-----------------------------------
Command:
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ollama" \
  -d '{
    "model": "qwen:14b",
    "messages": [{"role": "user", "content": "What is the largest planet in our solar system?"}]
  }'

Result: Successfully returned "The largest planet in our solar system is Jupiter."

STEP 8: ENABLE EXTERNAL ACCESS
------------------------------
Problem: Ollama was only listening on localhost (127.0.0.1:11434)

Solution Steps:

1. Check current listening status:
   Command: ss -tlnp | grep 11434
   Result: LISTEN on 127.0.0.1:11434 only

2. Create systemd override directory:
   Command: sudo mkdir -p /etc/systemd/system/ollama.service.d/

3. Create override configuration:
   Command: sudo nano /etc/systemd/system/ollama.service.d/override.conf
   Content:
   [Service]
   Environment="OLLAMA_HOST=0.0.0.0"

4. Reload systemd and restart service:
   Commands:
   sudo systemctl daemon-reload
   sudo systemctl restart ollama

5. Verify new listening status:
   Command: ss -tlnp | grep 11434
   Result: LISTEN on *:11434 (all interfaces)

6. Test external access:
   Command: curl http://10.1.1.198:11434/api/version
   Result: {"version":"0.9.0"}

CREATED FILES
-------------
1. /home/cslog/llm/ollama_api_example.py
   - Python script demonstrating native and OpenAI-compatible API usage
   - Shows streaming and non-streaming examples

2. /home/cslog/llm/ollama_api_curl_examples.sh
   - Shell script with curl command examples
   - Quick reference for API testing

3. /home/cslog/llm/enable_external_access.sh
   - Instructions for enabling external access
   - Security considerations and firewall rules

4. /home/cslog/llm/external_api_usage.md
   - Usage examples for various programming languages
   - Integration with OpenAI clients, LangChain, etc.

API ENDPOINTS
-------------
Base URL (local): http://localhost:11434
Base URL (external): http://10.1.1.198:11434

Main Endpoints:
- Native API: /api/generate
- OpenAI-compatible: /v1/chat/completions
- List models: /api/tags
- Model info: /api/show

USAGE EXAMPLES
--------------
1. With OpenAI Python library:
   from openai import OpenAI
   client = OpenAI(base_url="http://10.1.1.198:11434/v1", api_key="ollama")
   response = client.chat.completions.create(model="qwen:14b", messages=[{"role": "user", "content": "Hello!"}])

2. With curl:
   curl -X POST http://10.1.1.198:11434/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer ollama" \
     -d '{"model": "qwen:14b", "messages": [{"role": "user", "content": "Hello!"}]}'

SECURITY NOTES
--------------
- The API is now accessible from any network interface
- No authentication is required (Authorization header can be any value)
- Consider implementing firewall rules for production use:
  sudo ufw allow from 10.1.1.0/24 to any port 11434

TROUBLESHOOTING
---------------
- If connection refused: Check systemctl status ollama
- If model not found: Run ollama list to see available models
- If GPU not used: Check nvidia-smi while model is running
- For logs: journalctl -u ollama -f

STEP 9: INSTALL WEB INTERFACE (OPEN WEBUI)
-------------------------------------------
Problem: Needed a user-friendly web interface for chatting with the model

Solution Steps:

1. Install Docker:
   Command: curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh
   Result: Docker installed successfully

2. Add user to docker group:
   Command: sudo usermod -aG docker $USER
   Result: User added to docker group

3. Install Docker Compose:
   Command: sudo apt-get install docker-compose-plugin
   Result: Docker Compose installed

4. Deploy Open WebUI container:
   Command:
   docker run -d \
     -p 3000:8080 \
     --add-host=host.docker.internal:host-gateway \
     -v open-webui:/app/backend/data \
     --name open-webui \
     --restart always \
     ghcr.io/open-webui/open-webui:main
   
   Result: Container deployed and running

5. Verify container status:
   Command: docker ps
   Result: open-webui container running on port 3000

6. Access Web Interface:
   URL: http://10.1.1.198:3000
   Note: First user to register becomes admin

WEB INTERFACE FEATURES
----------------------
- Modern chat interface similar to ChatGPT
- Conversation history and management
- Model selection dropdown
- Dark/light theme support
- Markdown and LaTeX rendering
- File upload capabilities
- Voice input/output support
- PWA support for mobile devices
- Automatic connection to Ollama at http://10.1.1.198:11434

DOCKER MANAGEMENT COMMANDS
--------------------------
- View logs: docker logs open-webui
- Stop container: docker stop open-webui
- Start container: docker start open-webui
- Remove container: docker rm open-webui
- Update container: 
  docker pull ghcr.io/open-webui/open-webui:main
  docker stop open-webui
  docker rm open-webui
  [Run the deploy command again]

ACCESS POINTS SUMMARY
---------------------
- Ollama API: http://10.1.1.198:11434
- Web Chat Interface: http://10.1.1.198:3000
- Both services are accessible from external machines on the network

PERFORMANCE
-----------
- Model size: 8.2GB
- VRAM usage: ~15GB when loaded
- Response time: <1 second for simple queries
- Supports concurrent requests

This setup provides a fully functional local LLM with both API access and a modern web interface,
compatible with OpenAI's API format and easy to use for both developers and end users.