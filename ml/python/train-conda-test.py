import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "Qwen/Qwen2.5-Coder-3B"

# 1) 4-bit bitsandbytes config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# 2) Load 4-bit model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    device_map="auto",
)

# 3) Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

# 4) Define LoRA config and wrap
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# 5) Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# A tiny dummy dataset
train_texts = [
    {
        "input_ids": torch.tensor(tokenizer.encode("Hello Qwen", add_special_tokens=False)),
        "labels":    torch.tensor(tokenizer.encode("Hello Qwen", add_special_tokens=False))
    }
]

class TinyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

train_dataset = TinyDataset(train_texts)

def collator(batch):
    input_ids = [b["input_ids"] for b in batch]
    labels    = [b["labels"]    for b in batch]

    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels,
        batch_first=True,
        padding_value=-100
    )
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask
    }

training_args = TrainingArguments(
    output_dir="./test_qwen_lora",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    fp16=True,
    gradient_checkpointing=True,
    logging_steps=1,
    max_steps=1,  # just do 1 step
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=collator,
    tokenizer=tokenizer,
)

trainer.train()
print("Training finished!")
