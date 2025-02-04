import glob
import os
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, PeftModel

# -------------------------
# 1. Gather .pas code chunks
# -------------------------
directories = ['ro', 'src']
examples = []
encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
base_path = "/path/code"

for directory in directories:
    dir_path = os.path.join(base_path, directory)
    pas_files = glob.glob(os.path.join(dir_path, "**/*.pas"), recursive=True)
    print(f"\nProcessing {directory} directory: found {len(pas_files)} .pas files")
    for file_path in pas_files:
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    code_text = f.read()
                    lines = code_text.splitlines()
                    chunk_size = 50  # or 100, etc.
                    for i in range(0, len(lines), chunk_size):
                        chunk = "\n".join(lines[i : i + chunk_size])
                        examples.append(chunk)
                break
            except UnicodeDecodeError:
                # Try next encoding
                continue

print(f"\nTotal number of chunks created: {len(examples)}")

# -------------------------
# 2. Create raw HF Dataset
# -------------------------
raw_dataset = Dataset.from_dict({"text": examples})

# -------------------------
# 3. Load Qwen tokenizer (official approach)
# -------------------------
model_name = "Qwen/Qwen2.5-Coder-3B"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    use_fast=True,
)

# -------------------------
# 4. Tokenize function: next-token (causal) objective
# -------------------------
def tokenize_function(batch):
    tokenized = tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=1024
    )
    # For causal LM, we typically copy the same input_ids to 'labels'
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# Tokenize entire dataset
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True)

# -------------------------
# 5. Data collator for causal LM
# -------------------------
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# -------------------------
# 6. Load base model and wrap with LoRA
# -------------------------
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    load_in_4bit=True,      # optional quantization
    device_map="auto",      # or "cuda:0"
    attn_implementation="flash_attention_2"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
peft_model = get_peft_model(base_model, lora_config)

# -------------------------
# 7. Training settings
# -------------------------
training_args = TrainingArguments(
    output_dir="./qwen_lora_out",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    logging_steps=10,
    num_train_epochs=1,
    bf16=True,     # if your GPU supports bf16
    save_steps=500,
)

# -------------------------
# 8. Trainer
# -------------------------
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator,
    tokenizer=tokenizer
)

trainer.train()  # optionally resume_from_checkpoint="..."
trainer.save_model()  # This saves the LoRA adapter weights by default

# -------------------------
# 9. Merge
# -------------------------

# 3. Merge the LoRA adapters into the base model
merged_model = peft_model.merge_and_unload()

# 4. Save the merged model as a normal Hugging Face checkpoint
merged_model.save_pretrained("./qwen_merged_model")

