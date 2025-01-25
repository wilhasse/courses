# Introduction

llama.cpp  
https://github.com/ggerganov/llama.cpp

# Build

```bash
git clone https://github.com/ggerganov/llama.cpp

cd llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release
```

# Run

Download the model, example:

```bash
wget https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf
```

Run prompt:

```bash
./llama.cpp/build/bin/llama-cli --model DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --n-gpu-layers 80 --threads 24 --cache-type-k q8_0 --prompt '<｜User｜>What is 1+1?<｜Assistant｜>' -no-cnv
```

Serve as API:

```bash
./llama.cpp/build/bin/llama-server --model DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf --n-gpu-layers 80 --threads 24 --batch-size 512 --cache-type-k q8_0 --port 8080 --host 0.0.0.0
```

Test it:

```bash
curl http://10.1.0.16:8080/completion -d '{
  "prompt": "<｜User｜>What is 1+1?<｜Assistant｜>",
  "n_predict": 100,
  "stop": ["<｜end▁of▁sentence｜>"]
}'
```
