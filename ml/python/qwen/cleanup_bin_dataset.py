import argparse
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO)

IGNORE_INDEX = -100

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to the raw dataset (e.g. .npy)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Where to save the cleaned dataset (e.g. .npy)")
    parser.add_argument("--tokenizer_path", type=str, required=True,
                        help="Path to the tokenizer to check for invalid tokens")
    return parser.parse_args()

def remove_invalid_tokens_from_example(example, tokenizer):
    vocab_plus_special = len(tokenizer)
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0

    replaced_count = 0

    def clean_ids(ids):
        nonlocal replaced_count
        new_ids = []
        for tid in ids:
            if tid < 0 or tid >= vocab_plus_special:
                new_ids.append(unk_id)
                replaced_count += 1
            else:
                new_ids.append(tid)
        return new_ids

    # Adjust these keys if your dataset’s keys differ
    example["input_ids"] = clean_ids(example["input_ids"])
    example["label"]     = clean_ids(example["label"])
    return example, replaced_count

def main():
    args = parse_args()
    logging.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    # 1) Load your raw dataset
    logging.info(f"Loading dataset from {args.input_path} ...")
    data = np.load(args.input_path, allow_pickle=True)
    # 'data' is presumably a list (or array) of dicts, each with "input_ids" and "label".

    logging.info(f"Dataset length: {len(data)} examples.")

    # 2) Clean up out-of-range tokens
    num_examples = len(data)
    num_fixed_examples = 0
    num_replaced_tokens = 0

    for i in tqdm(range(num_examples), desc="Cleaning dataset"):
        ex = data[i]
        # We expect "input_ids" and "label" in ex
        old_input_ids = ex["input_ids"]
        old_label     = ex["label"]

        new_ex, replaced_count = remove_invalid_tokens_from_example(ex, tokenizer)
        data[i] = new_ex  # store back

        if replaced_count > 0:
            num_fixed_examples += 1
            num_replaced_tokens += replaced_count

    # Info
    logging.info(
        f"Finished cleaning: replaced {num_replaced_tokens} invalid tokens "
        f"in {num_fixed_examples} examples."
    )
    logging.info(f"Total {num_examples} examples.")

    # 3) Save the cleaned dataset
    logging.info(f"Saving cleaned dataset to {args.output_path}")
    np.save(args.output_path, data, allow_pickle=True)

    # If you'd like to compress, you can do something like:
    # np.savez_compressed(args.output_path + ".npz", data=data)
    # though you then need to load it with np.load(...)


if __name__ == "__main__":
    main()
