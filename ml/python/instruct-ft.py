import torch
import json
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel   
from transformers import Trainer
from datasets import Dataset

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq
)

from peft import (
    LoraConfig, 
    get_peft_model, 
    get_peft_model_state_dict, 
    prepare_model_for_kbit_training
)

from transformers import BitsAndBytesConfig   
model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # or load_in_4bit=True,
    llm_int8_threshold=6.0,  # optional
    llm_int8_has_fp16_weight=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    quantization_config=bnb_config,  # or bitsandbytes_config in older versions
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

from datasets import Dataset

def load_from_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)  # Load the JSON array directly
    return Dataset.from_dict({"data": raw_data})

# Usage
dataset = load_from_json("/home/cslog/finetune_dataset_all_crud.json")

def create_prompt(example):
    # Minimal Qwen chat style
    # You may include a system prompt if you prefer:
    system_text = (
        "<|im_start|>system\n"
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n"
        "<|im_end|>\n"
    )

    user_text = f"<|im_start|>user\n{example['instruction']}\n<|im_end|>\n"
    assistant_text = f"<|im_start|>assistant\n{example['output']}\n<|im_end|>"

    return system_text + user_text + assistant_text

def format_dataset(batch):
    return {"text": [create_prompt(x) for x in batch["data"]]}

dataset = dataset.map(format_dataset, batched=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def tokenize_and_add_labels(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        max_length=1024,
        padding="max_length"
    )
    # Copy input_ids to labels for causal LM
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize_and_add_labels, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8) 
# pad_to_multiple_of=8 is often more efficient on GPUs

def compute_loss_for_qwen(model, inputs):
    """
    Custom loss function for Qwen-like code models
    which only return `logits`. We manually apply a
    causal language modeling loss.
    """
    # Grab input_ids, attention_mask, labels from the batch
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask", None)
    labels = inputs["labels"]  # This must already be aligned (-100 for ignored tokens)

    # Forward pas
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        # Some Qwen variants do accept `labels=labels`, but if it doesn't return loss, we do it manually:
    )
    logits = outputs.logits  # shape: (batch_size, sequence_length, vocab_size)

    # Shift so that tokens <n> predict token <n+1>
    # typical "causal language modeling" shift
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100  # usual for padded tokens
    )
    return loss

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="qwen-lora-out",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    #fp16=False,
    learning_rate=2e-4,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=50,
    logging_steps=10,
    optim="adamw_torch",
    report_to="none",
)

class QwenTrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        **kwargs,  # capture extra parameters
    ):
        import torch
        import torch.nn.functional as F

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        labels = inputs["labels"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # standard causal LM shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # return either just loss or (loss, outputs)
        if return_outputs:
            return (loss, outputs)
        return loss

trainer = QwenTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)
trainer.train()

# After trainer.train() finishes
trainer.save_model("my-lora-checkpoints")

base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"  # or your original base
lora_output_dir = "my-lora-checkpoints"            # your save folder

# Load the *base* model in 4bit/8bit or full precision
# as you did for training
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    trust_remote_code=True,
    device_map="auto",
    load_in_8bit=True,   # or load_in_4bit, or normal fp16
)

# Then load the LoRA adapter
model = PeftModel.from_pretrained(
    model,
    lora_output_dir,
    # If using 8-bit or 4-bit training, pass these too:
    # is_trainable=False (some versions)
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_name,
    trust_remote_code=True
)

# evaluation    
model.eval()

# Test
# Example prompt (system + user) in Qwen format
system_text = (
    "<|im_start|>system\n"
    "You are Qwen. Please answer helpfully.\n"
    "<|im_end|>\n"
)
user_text = (
    "<|im_start|>user\n"
    "Explain how code builds and executes a SQL query using sql_s, crit, and the statement SELECT * FROM VERSAO WHERE VERSAO = 2.'\n"
    "<|im_end|>\n"
)
prompt = system_text + user_text + "<|im_start|>assistant\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # or device_map
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.8,
    )

# Decode the result. For Qwen chat style, it might put answer before <|im_end|>.
result_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(result_text)