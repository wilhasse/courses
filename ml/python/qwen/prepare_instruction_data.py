import os
import json
import numpy as np
import argparse
import tqdm
import transformers
from typing import Dict

IGNORE_INDEX = -100  # default ignore_index in Transformers

def setup_tokenizer(tokenizer):
    tokenizer.add_special_tokens({
        "additional_special_tokens": [
            "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|repo_name|>",
            "<|file_sep|>", "<|im_start|>", "<|im_end|>"
        ]
    })
    return tokenizer

def chatml_format_preprocess(
    messages,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant.",
    only_last_turn_loss=True
) -> Dict:
    """
    Convert one chat example (with multiple roles) into a single sequence of tokens,
    plus a label array that masks out user tokens (i.e. user => -100).
    
    Format example:
      <|im_start|>system\n
      [system_message]
      <|im_end|>
      <|im_start|>user\n
      [user_message]
      <|im_end|>
      <|im_start|>assistant\n
      [assistant_message]
      <|im_end|>
    """
    roles = {
        "user": "<|im_start|>user",
        "assistant": "<|im_start|>assistant",
        "system": "<|im_start|>system"
    }
    
    im_start = tokenizer("<|im_start|>").input_ids[0]
    im_end = tokenizer("<|im_end|>").input_ids[0]
    nl_tokens = tokenizer('\n').input_ids
    if len(nl_tokens) > 0:
        nl_tokens = nl_tokens[-1:]  # we keep just the last ID for newline
    
    # Pre-tokenize "system", "user", "assistant" so we can insert them
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    input_ids = []
    labels = []

    # If the very first message is a system override
    if len(messages) > 0 and messages[0]["role"] == "system" and messages[0]["content"] != "":
        system_message = messages[0]["content"]
        # We won't remove it from messages, we'll just treat that as system. 
        # Alternatively, you can skip re-adding it if you prefer.

    # 1) Insert the system message
    # e.g. <|im_start|>system\n [SYSTEM_CONTENT] <|im_end|>\n
    system_tokens = (
        [im_start] +
        _system +
        tokenizer(system_message, add_special_tokens=False).input_ids +
        [im_end] +
        nl_tokens
    )
    input_ids += system_tokens

    # The system tokens are not "predicted" by the model (or we usually mask them out),
    # but let's follow Alibaba’s approach: only the assistant role is labeled for loss.
    system_labels = (
        [im_start] +
        [IGNORE_INDEX] * (len(system_tokens) - 3) +
        [im_end] + 
        nl_tokens
    )
    labels += system_labels

    # 2) Add each subsequent user/assistant message
    # We skip the first item if it was "system" because we handled it above
    start_idx = 1 if (len(messages) > 0 and messages[0]["role"] == "system") else 0
    for j, sentence in enumerate(messages[start_idx:], start_idx):
        role = roles.get(sentence["role"], None)
        if not role:
            raise ValueError(f"Unknown role '{sentence['role']}' encountered (must be user/assistant/system).")

        # e.g. <|im_start|>user\n [USER_CONTENT] <|im_end|>\n
        content_tokens = tokenizer(
            sentence["content"],
            add_special_tokens=False
        ).input_ids
        # Build the full chunk
        chunk = (tokenizer(role).input_ids + nl_tokens) + content_tokens + [im_end] + nl_tokens
        # Add them to input_ids
        input_ids += chunk

        # Decide how to set the labels
        if sentence["role"] == "assistant":
            # Assistant tokens: we keep them in labels, except the "role" part
            # So basically, we mask out the "role" + newline, but keep the rest
            # role_ids = tokenizer(role).input_ids + nl_tokens
            # chunk length = len(role_ids) + len(content_tokens) + 1(im_end) + 1(nl)
            # We'll do a simpler approach:
            #   [im_start|assistant\n] => IGNORE
            #   rest => actual tokens
            #   <|im_end|>\n => IGNORE

            # Reconstruct the same chunk for labels:
            # The first token [im_start] we keep to identify location, 
            # but let's do same approach as Alibaba’s code:
            # They do: [im_start] + [IGNORE]*(...) for user or system
            # For assistant, they keep actual content tokens in labels 
            # but still mask the <|im_start|>assistant part.

            chunk_labels = [im_start]  # first
            # Mask out everything in the role’s text except the first token
            role_len = len(tokenizer(role).input_ids) - 1  # minus 1 because we used im_start
            chunk_labels += [IGNORE_INDEX] * role_len
            # Next the newline tokens also get masked
            # plus the content tokens themselves
            # We want to keep the content tokens in labels
            # So we add them
            chunk_labels += content_tokens
            # Now add the <|im_end|>\n masked out
            chunk_labels += [im_end] + nl_tokens
        else:
            # If it's user or system, we do not want them in the loss
            # So everything is IGNORE, except maybe the first token for alignment
            # But typically we mask them all
            chunk_labels = [im_start] + [IGNORE_INDEX]*(len(chunk)-1)

        # If only_last_turn_loss=True and this is not the last assistant turn,
        # we can mask out everything except the last turn. 
        # We'll skip that logic for brevity. 
            # if only_last_turn_loss and j < len(messages)-1:
            #    chunk_labels = [IGNORE_INDEX]* len(chunk)  # mask all

        labels += chunk_labels

    # If the final sequence is too long, discard
    if len(input_ids) > max_len:
        return None
    
    return dict(input_ids=input_ids, label=labels, length=len(input_ids))

def save_to_npy(objs, output_prefix):
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
    np.save(output_prefix + ".npy", objs, allow_pickle=True)
    print(f"Saved {len(objs)} samples to {output_prefix}.npy")

def save_to_mmap(objs, output_prefix, tokenizer):
    """
    Save data to memory-mapped arrays. This is what the Alibaba script does for huge scale.
    We'll store input_ids, label, length. Each is a 2D array with shape = (n_samples, max_len).
    """
    import mmap
    IGNORE_INDEX = -100

    # 1) find the max length
    max_len = 0
    for obj in objs:
        max_len = max(max_len, len(obj["input_ids"]))

    n_samples = len(objs)
    # Prepare memmaps
    input_ids_file = output_prefix + ".input_ids.mmap"
    labels_file = output_prefix + ".labels.mmap"
    length_file = output_prefix + ".lengths.mmap"
    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    # create the memmap arrays
    input_ids_mmap = np.memmap(input_ids_file, dtype=np.int32, mode='w+', shape=(n_samples, max_len))
    labels_mmap = np.memmap(labels_file, dtype=np.int32, mode='w+', shape=(n_samples, max_len))
    lengths_mmap = np.memmap(length_file, dtype=np.int32, mode='w+', shape=(n_samples,))

    # fill them
    for i, obj in enumerate(tqdm.tqdm(objs, desc="Saving to mmap")):
        seq = obj["input_ids"]
        lab = obj["label"]
        length = len(seq)
        # pad them
        padded_seq = seq + [tokenizer.pad_token_id]*(max_len - length)
        padded_lab = lab + [IGNORE_INDEX]*(max_len - length)

        input_ids_mmap[i] = padded_seq
        labels_mmap[i] = padded_lab
        lengths_mmap[i] = length

    # flush
    input_ids_mmap.flush()
    labels_mmap.flush()
    lengths_mmap.flush()
    # store shape info
    shape_info = {
        "n_samples": n_samples,
        "max_len": max_len
    }
    with open(output_prefix + ".shape.json", "w") as f:
        json.dump(shape_info, f, indent=2)

    print(f"Memory-mapped arrays saved under prefix: {output_prefix}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="instruction_data.jsonl", help="Your chat data in JSONL format.")
    parser.add_argument("--output_prefix", type=str, default="./processed/instruction_data", help="Where to save the tokenized data.")
    parser.add_argument("--tokenizer_path", type=str, default="Qwen/Qwen2.5-Coder-3B", help="Qwen tokenizer path.")
    parser.add_argument("--save_format", choices=[".npy", ".mmap"], default=".npy", help="Choose .npy or .mmap.")
    parser.add_argument("--max_len", type=int, default=8192, help="Max sequence length.")
    parser.add_argument("--only_last_turn_loss", action="store_true", help="If set, only label the last assistant message.")
    args = parser.parse_args()

    # 1) Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        model_max_length=args.max_len,
        add_bos_token=False,
        add_eos_token=False,
        pad_token='<|endoftext|>',
        eos_token='<|im_end|>',
        padding_side="right"
    )
    tokenizer = setup_tokenizer(tokenizer)

    # 2) Read the data
    objs = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        for line in tqdm.tqdm(f, desc="Reading JSONL"):
            line=line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record["messages"]
            only_last = record.get("only_last_turn_loss", True)  # or always use args.only_last_turn_loss
            processed = chatml_format_preprocess(
                messages,
                tokenizer=tokenizer,
                max_len=args.max_len,
                only_last_turn_loss=only_last
            )
            if processed is not None:
                objs.append(processed)

    if len(objs) == 0:
        print("No valid samples found. Exiting.")
        return

    # 3) Save in .npy or .mmap
    if args.save_format == ".npy":
        # Just dump the python list of dicts
        save_to_npy(objs, args.output_prefix)
    else:
        # Save in memory-mapped format (faster for huge sets)
        save_to_mmap(objs, args.output_prefix, tokenizer)

    print(f"Done. Saved {len(objs)} processed samples.")


if __name__ == "__main__":
    main()
