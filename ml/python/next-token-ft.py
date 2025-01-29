import glob
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

#
# Read source code
#
files = glob.glob("/path/to/source/**/*.pas", recursive=True)
examples = []

# List of encodings to try
encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

for file_path in files:
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                code_text = f.read()
                lines = code_text.splitlines()
                chunk_size = 50  # lines per chunk
                for i in range(0, len(lines), chunk_size):
                    chunk = "\n".join(lines[i : i + chunk_size])
                    examples.append(chunk)
            break  # If successful, break the encoding loop
        except UnicodeDecodeError:
            if encoding == encodings[-1]:  # If this was the last encoding to try
                print(f"Failed to decode {file_path} with any encoding")
            continue

# print
print(f"Total number of files processed: {len(files)}")
print(f"Total number of chunks created: {len(examples)}")
print(f"Average chunks per file: {len(examples)/len(files):.2f}")

# model
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 4. Tokenize samples
tokenizer = AutoTokenizer.from_pretrained(model_name)  # Example name
def tokenize_function(examples):
    return tokenizer(examples, padding="max_length", truncation=True, max_length=1024)

# Convert the list of code chunks into a dataset
from datasets import Dataset
raw_dataset = Dataset.from_dict({"text": examples})
tokenized_dataset = raw_dataset.map(lambda x: tokenize_function(x["text"]), batched=True)

#
# Tokenize
#
def tokenize_function(texts):
    # For causal LM, we often do something like:
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=1024
    )
    # Copy input_ids to labels so the model computes cross-entropy loss
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # For causal language modeling (GPT-style)
)

# 5. Setup LoRA config
base_model = AutoModelForCausalLM.from_pretrained(model_name,load_in_4bit=True,device_map="auto")

# Lora config
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
peft_model = get_peft_model(base_model, lora_config)

#
# Training
#
training_args = TrainingArguments(
    output_dir="./qwen_lora_out",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=1,
    fp16=True,
    save_steps=500,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,  # <--- important
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model()