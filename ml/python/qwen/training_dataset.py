import json
from transformers import AutoTokenizer
from datasets import Dataset

IGNORE_INDEX = -100

def chatml_format_preprocess(messages, tokenizer, max_len=2048, only_last_turn_loss=True):
    """
    Splits the conversation into system/user/assistant chunks.
    Then tokenizes each chunk and applies -100 where appropriate.
    """

    IGNORE_INDEX = -100

    roles = {
        "user": "<|im_start|>user\n",
        "assistant": "<|im_start|>assistant\n",
        "system": "<|im_start|>system\n",
    }
    im_end_token = "<|im_end|>\n"

    # If first is system, use it; else fallback
    system_message = ""
    if messages and messages[0]["role"] == "system":
        system_message = messages[0]["content"]
        msg_index_start = 1
    else:
        system_message = "You are a helpful assistant."
        msg_index_start = 0

    # Tokenize the system message chunk
    system_chunk_text = f"{roles['system']}{system_message}{im_end_token}"
    system_chunk = tokenizer(system_chunk_text, add_special_tokens=False)

    # We'll accumulate final input_ids and labels across all chunks
    final_input_ids = []
    final_labels = []

    # For system chunk, we always mask because we usually do not want to train on system text
    final_input_ids.extend(system_chunk["input_ids"])
    final_labels.extend([IGNORE_INDEX] * len(system_chunk["input_ids"]))

    # Identify all assistant segments (in case we only keep the last one)
    assistant_indices = []
    # We’ll separately store each tokenized chunk so we can unify them at the end
    chunk_store = []

    for i, msg in enumerate(messages[msg_index_start:]):
        role = msg["role"]
        content = msg["content"]

        if role not in roles:
            raise ValueError(f"Unknown role: {role}")

        # Build raw text for this chunk
        chunk_text = f"{roles[role]}{content}{im_end_token}"
        chunk_tokens = tokenizer(chunk_text, add_special_tokens=False)

        # Save whether this chunk is user or assistant, plus the tokenized IDs
        is_asst = (role == "assistant")
        chunk_store.append((chunk_tokens["input_ids"], is_asst))

        if is_asst:
            assistant_indices.append(len(chunk_store) - 1)

    if only_last_turn_loss:
        # We'll keep labels only for the *last* assistant chunk, mask out everything else.
        last_asst_idx = assistant_indices[-1] if assistant_indices else None

        for chunk_idx, (chunk_ids, is_asst) in enumerate(chunk_store):
            final_input_ids.extend(chunk_ids)
            if is_asst and (chunk_idx == last_asst_idx):
                # This is the last assistant => keep label tokens
                final_labels.extend(chunk_ids)
            else:
                # Mask out
                final_labels.extend([IGNORE_INDEX] * len(chunk_ids))

    else:
        # Keep labels for *all* assistant chunks; mask out user & system
        for chunk_idx, (chunk_ids, is_asst) in enumerate(chunk_store):
            final_input_ids.extend(chunk_ids)
            if is_asst:
                final_labels.extend(chunk_ids)
            else:
                final_labels.extend([IGNORE_INDEX] * len(chunk_ids))

    # Truncate if needed
    if len(final_input_ids) > max_len:
        final_input_ids = final_input_ids[:max_len]
        final_labels = final_labels[:max_len]

    return {
        "input_ids": final_input_ids,
        "labels": final_labels,
    }

# --------------------------
# Example usage
# --------------------------
def build_dataset_from_jsonl(jsonl_path, tokenizer, max_len=2048):
    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            messages = obj["messages"]
            only_last_turn_loss = obj.get("only_last_turn_loss", True)
            processed = chatml_format_preprocess(
                messages, tokenizer, max_len=max_len, only_last_turn_loss=only_last_turn_loss
            )
            if processed:
                samples.append(processed)

    print(f"\nTotal samples processed: {len(samples)}")
    return Dataset.from_list(samples)


# Then, in your main script:
if __name__ == "__main__":
    import os
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model

    # 1) Load tokenizer
    model_name = "Qwen/Qwen2.5-Coder-3B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        # Possibly also:
        # pad_token='<|endoftext|>',
        # eos_token='<|im_end|>',
        # etc., if Qwen requires them.
    )

    # 2) Build your dataset from a chat-based JSONL
    chat_dataset = build_dataset_from_jsonl("./my_instruct_data.jsonl", tokenizer, max_len=2048)

    print("\nDataset info:")
    print(chat_dataset)
    print("\nFirst example in dataset:")
    print(chat_dataset[0])

    # 3) (Optional) train/val split if you have a single file
    # chat_dataset = chat_dataset.train_test_split(test_size=0.05)

    # 4) Collator & training
    def data_collator(features):
        # If you already have "input_ids" and "labels", you usually just do
        # pad them. Here's a minimal approach:
        import torch
        batch_input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        batch_labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=IGNORE_INDEX)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(tokenizer.pad_token_id),
        }

    # 5) Load model & LoRA
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, trust_remote_code=True, load_in_4bit=True, device_map="auto"
    )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    peft_model = get_peft_model(base_model, lora_config)

    training_args = TrainingArguments(
        output_dir="./instruct_out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        logging_steps=10,
        bf16=True,
        save_steps=2000
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=chat_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    trainer.train()
    trainer.save_model()

    # Merge
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained("./instruct_merged_model")
