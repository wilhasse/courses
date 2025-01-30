# Code

- [Generate instruction dataset](./../python/dataset-claude.py)

- [Instruct model](./../python/instruct-ft.py)

- [Next Token fine-tuning](./../python/next-token-ft.py)

- [Merge Lora fine-tuned model](./../python/merge-model.py)

# Commands


Convert merged lora ft model to gguf

```bash
python3 ./llama.cpp/convert_hf_to_gguf.py \
    --outfile first-test-model.gguf \
    --outtype q8_0 \
    --verbose \
    ./first-test-model
```

Load on ollama

```bash
ollama create first-test-model -f Modelfile
ollama list
#NAME                       ID              SIZE      MODIFIED       
#first-test-model:latest    d129a5933cd7    15 GB     20 minutes ago    
```
