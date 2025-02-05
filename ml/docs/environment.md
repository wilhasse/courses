# Jupyter

Install

```bash
# install
pip install ipykernel jupyter jupyterlab

# Now register the kernel
python -m ipykernel install --user --name=miniconda3_base

jupyter notebook --generate-config
pico /home/cslog/.jupyter/jupyter_notebook_config.py

# uncomment and edit
c.NotebookApp.ip = '0.0.0.0'

# run
jupyter notebook --no-browser
```

# Packages

Select kernel: miniconda3_base

```bash
# base
!pip install pytorch transformers datasets peft bitsandbytes accelerate

# For 8-bit or 4-bit quantization (optional)
!pip install bitsandbytes

# For speedups, optionally install xformers
!pip install xformers

# or Flash Attention if supported
!pip install flash-attn
```

# Test

Test GPU

```python
import sys
print(sys.executable)  # Should show path to your miniconda installation

import torch
print(torch.__version__)
#/home/cslog/miniconda3/bin/python
#2.5.1+cu124

# Test if CUDA is available
if torch.cuda.is_available():
    print("CUDA Device:", torch.cuda.get_device_name(0))
    print("CUDA Version:", torch.version.cuda)
else:
    print("CUDA is not available")

#CUDA Device: NVIDIA GeForce RTX 4090
#CUDA Version: 12.4
```

# Other tools

Ollama  
https://github.com/ollama/ollama  

Open Webui  
https://github.com/open-webui/open-webui  

Install

```bash
curl -fsSL https://ollama.com/install.sh | sh
pip install open-webui
```

Ollama run

```bash
# prompt
ollama run qwen2.5

# api
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5",
  "prompt":"Hello"
}'
```

Run webui

```bash
open-webui serve
```

# Conda

```bash
# remove all
conda deactivate
conda remove -n qwen_sft --all

# new
conda create -n qwen_sft python=3.10 -y
conda activate qwen_sft
conda install cudatoolkit=11.8 -c nvidia -y
 
# pip install
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy<2
pip install --force-reinstall "bitsandbytes>=0.43.2"
pip install transformers>=4.31.0 accelerate>=0.21.0 peft>=0.4.0
pip install "numpy<2" --force-reinstall

# test
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
python -c "import numpy; print(numpy.__version__)"
python -m bitsandbytes

python train-conda-test.py
```