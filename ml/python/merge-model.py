import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
lora_output_dir = "my-lora-checkpoints"

# Optionally force CPU merge if you don’t want to disrupt your GPU training.
# Just ensure you have enough CPU RAM for the 7B model in float32!
device = torch.device("cpu")

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map=None,              # We'll put it on CPU
    torch_dtype=torch.float32,    # Merge in float32
    trust_remote_code=True
).to(device)

# 2. Load the LoRA adapter
model = PeftModel.from_pretrained(
    base_model,
    lora_output_dir,
    # In some versions you might also need: is_trainable=False
)

model.eval()

# Merge the LoRA weights into the base model
merged_model = model.merge_and_unload()

save_path = "my-merged-model"  # your output folder
merged_model.save_pretrained(save_path)

# Also save tokenizer, if needed:
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.save_pretrained(save_path)

model = AutoModelForCausalLM.from_pretrained(save_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(save_path, trust_remote_code=True)

# Done. The weights are already merged.
