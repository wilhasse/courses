# Available Qwen Models for RTX 4090 (24GB VRAM)

## Current Model
- **qwen:14b** (8.2GB) - Already installed

## Recommended Upgrades

### 1. Qwen3:32b (BEST CHOICE)
- **Size**: 20GB - Fits comfortably in 24GB VRAM
- **Command**: `ollama pull qwen3:32b`
- **Features**: Latest generation, 40K context window
- **Why**: 2.3x larger than current model, still leaves ~4GB headroom

### 2. Qwen2.5:32b-instruct
- **Size**: 20GB
- **Command**: `ollama pull qwen2.5:32b-instruct`
- **Features**: 128K context window, trained on 18 trillion tokens
- **Why**: Longer context window for processing larger documents

### 3. Qwen3:14b-q8_0 (Higher Quality)
- **Size**: 16GB
- **Command**: `ollama pull qwen3:14b-q8_0`
- **Features**: Same parameter count but higher precision (8-bit vs 4-bit)
- **Why**: Better quality responses than standard quantization

## Model Comparison

| Model | Parameters | Size | Context | Generation |
|-------|------------|------|---------|------------|
| qwen:14b (current) | 14B | 8.2GB | 8K | Qwen 1.5 |
| qwen3:14b | 14B | 9.3GB | 40K | Qwen 3 |
| qwen3:32b | 32B | 20GB | 40K | Qwen 3 |
| qwen2.5:32b-instruct | 32B | 20GB | 128K | Qwen 2.5 |

## Too Large for 24GB VRAM
- qwen3:32b-q8_0 (35GB)
- qwen2.5:72b models (40GB+)
- qwen3:235b-a22b MoE models

## Quick Test Commands

```bash
# Pull the recommended 32B model
ollama pull qwen3:32b

# Test it
echo "Explain quantum computing in simple terms" | ollama run qwen3:32b

# Compare with current model
echo "Explain quantum computing in simple terms" | ollama run qwen:14b

# Check GPU usage
nvidia-smi
```

## Switching Between Models

The web interface at http://10.1.1.198:3000 will automatically detect all installed models and let you switch between them in the model dropdown.