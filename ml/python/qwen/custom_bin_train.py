import argparse
import logging
import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import BitsAndBytesConfig
from tqdm import tqdm

import transformers
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
)

# PEFT for LoRA
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Qwen's dataset utilities
from utils.training_datasets import SupervisedDataset, MMAPSupervisedDataset

logging.basicConfig(level=logging.INFO)

IGNORE_INDEX = -100

@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="/home/cslog/Qwen2.5-Coder-3B")
    use_flash_attention: bool = field(default=False)

@dataclass
class DataArguments:
    data_path: str = field(default="/home/cslog/sft_output.jsonl.npy")

@dataclass
class MyTrainingArguments(TrainingArguments):
    model_max_length: int = field(default=2048)
    use_peft: bool = field(default=True)
    truncate_source: bool = field(default=True)

    # This ensures custom fields are retained correctly
    def __post_init__(self):
        super().__post_init__()  # important so HF doesn't strip custom fields

def make_supervised_data_module(tokenizer, data_args, training_args) -> dict:
    data_path = data_args.data_path

    if data_path.endswith(".npy") or data_path.endswith(".jsonl"):
        train_dataset = SupervisedDataset(
            data_path=data_path,
            tokenizer=tokenizer,
            args=training_args
        )
    elif data_path.endswith(".mmap"):
        train_dataset = MMAPSupervisedDataset(
            tokenizer=tokenizer,
            data_path=data_path,
            args=None
        )
    else:
        raise ValueError("data_path must be .npy, .jsonl, or .mmap")
    
    # 3) Data collator
    def data_collator(batch):
        input_ids = [torch.flip(torch.tensor(x["input_ids"]), dims=[0]) for x in batch]
        labels = [torch.flip(torch.tensor(x["labels"]), dims=[0]) for x in batch]

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=tokenizer.pad_token_id
        ).flip(dims=[1])

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        ).flip(dims=[1])

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(tokenizer.pad_token_id)
        }

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def run_single_gpu_train():
    model_args = ModelArguments()
    data_args = DataArguments()
    training_args = MyTrainingArguments(
        output_dir="./qwen_bugfix_lora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=50,
        save_steps=200,
        evaluation_strategy="no",
        save_total_limit=1,
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        truncate_source=True  # explicitly set if needed
    )

    # 1) Load model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2" if model_args.use_flash_attention else None,
        use_cache=False  # Required for gradient checkpointing
    )

    # 2) Prepare for PEFT
    model = prepare_model_for_kbit_training(model)

    # 3) Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        bias="none"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4) Configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        model_max_length=training_args.model_max_length,
        # We'll let the base vocab load first, then add tokens if needed below
        padding_side="left"
    )

    # Instead of:
    logging.info("tokenizer.vocab_size:", tokenizer.vocab_size)
    logging.info("len(tokenizer):", len(tokenizer))
    logging.info("tokenizer.all_special_tokens:", tokenizer.all_special_tokens)

    # Use any of these correct approaches:

    # Option 1 - Using f-strings (recommended):
    logging.info(f"tokenizer.vocab_size: {tokenizer.vocab_size}")
    logging.info(f"len(tokenizer): {len(tokenizer)}")
    logging.info(f"tokenizer.all_special_tokens: {tokenizer.all_special_tokens}")

    # Option 2 - Using % formatting:
    logging.info("tokenizer.vocab_size: %d", tokenizer.vocab_size)
    logging.info("len(tokenizer): %d", len(tokenizer))
    logging.info("tokenizer.all_special_tokens: %s", tokenizer.all_special_tokens)

    # Option 3 - Using string concatenation:
    logging.info("tokenizer.vocab_size: " + str(tokenizer.vocab_size))
    logging.info("len(tokenizer): " + str(len(tokenizer)))
    logging.info("tokenizer.all_special_tokens: " + str(tokenizer.all_special_tokens))

    logging.info("")
    logging.info(f"Checking special tokens ...")

    # Some list of custom tokens you use (e.g. fim tokens, etc.)
    custom_special_tokens = [
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
        "<|repo_name|>",
        "<|file_sep|>",
        "<|im_start|>",
        "<|im_end|>"
    ]

    # We'll gather any tokens that aren't already in the vocabulary
    tokens_to_add = []
    for tok in custom_special_tokens:
        if tok not in tokenizer.get_vocab():
            tokens_to_add.append(tok)

    # If there's anything new, add them as additional special tokens
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
        logging.info(f"Added special tokens: {tokens_to_add}")

        # Resize model embeddings to match the new vocab size
        model.resize_token_embeddings(len(tokenizer))
 
    logging.info(f"Preparing Dataset ...")

    # 5) Prepare data module
    data_module = make_supervised_data_module(tokenizer, data_args, training_args)

    # 6) Create and configure trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    # 7) Train
    trainer.train()
    trainer.save_state()
    trainer.save_model()


if __name__ == "__main__":
    # For debugging device asserts, you can run:
    #    CUDA_LAUNCH_BLOCKING=1 python my_train.py
    run_single_gpu_train()
