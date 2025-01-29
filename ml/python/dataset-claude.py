import os
import re
import json
import anthropic

#------------------------------------------------------------------------------
# CONFIGURATION
#------------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# Directory containing your .pas or .sql or other source files
SOURCE_CODE_DIR = ""
# Output JSON file for your fine-tuning data
OUTPUT_JSON_FILE = "finetune_dataset.json"

MODEL_NAME = "claude-3-5-sonnet-latest"
MAX_TOKENS = 512
TEMPERATURE = 0.7

# Prompt guiding Anthropic to produce a JSON with {instruction, output}
INSTRUCTION_PROMPT = (
    "You are helping create a dataset to fine-tune a model on our custom code pattern.\n\n"
    "Here is a relevant code block:\n\n"
    "```{snippet}```\n\n"
    "1) Generate a short 'instruction' about this snippet.\n"
    "2) Provide a thorough 'output' explaining or improving it.\n\n"
    "Return valid JSON like:\n"
    "{{\n"
    '  "instruction": "Explain the SQL usage",\n'
    '  "output": "Here is what is happening in detail..." \n'
    "}}\n"
    "Now please provide your JSON."
)

def main():
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # 1) Gather relevant .pas or .sql files
    code_filepaths = []
    for root, _, files in os.walk(SOURCE_CODE_DIR):
        for filename in files:
            if filename.lower().endswith((".pas", ".sql")):
                code_filepaths.append(os.path.join(root, filename))

    # We'll store all code blocks here
    all_snippets = []
    snippet_count = 0
    max_snippets = 200  # limit how many total for cost reasons

    # 2) Extract multi-line blocks for sql_s, sql_i, sql_u, sql_d
    #    We'll consider 'crit.Clear' as well if relevant
    start_pattern = re.compile(r"(crit\.Clear;|sql_[siud]\.Clear;)", re.IGNORECASE)
    end_pattern   = re.compile(r"(sql_[siud]\.Execute)", re.IGNORECASE)

    for filepath in code_filepaths:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.read().splitlines()

        start_idx = -1
        for i, line in enumerate(lines):
            # If we see e.g. 'crit.Clear;' or 'sql_s.Clear;' or 'sql_i.Clear;' etc.
            if start_pattern.search(line) and start_idx == -1:
                start_idx = i

            # If we see e.g. 'sql_s.Execute' or 'sql_i.Execute' or 'sql_d.Execute' etc.
            if end_pattern.search(line) and start_idx != -1:
                snippet_block = lines[start_idx : i + 1]
                snippet_block_str = "\n".join(snippet_block).strip()
                if snippet_block_str:
                    all_snippets.append(snippet_block_str)
                    snippet_count += 1
                start_idx = -1

                # If we've reached our max snippet limit, break out early
                if snippet_count >= max_snippets:
                    break
        if snippet_count >= max_snippets:
            break

    print(f"Found {len(all_snippets)} multi-line code blocks that match [sql_s|sql_i|sql_u|sql_d].")

    # 3) For each block, call Anthropic to generate {instruction, output}
    fine_tune_data = []
    for idx, snippet_block in enumerate(all_snippets, start=1):
        print(f"\nSending block #{idx} to Claude...\n")

        # Build the user message
        messages = [
            {
                "role": "user",
                "content": INSTRUCTION_PROMPT.format(snippet=snippet_block)
            }
        ]

        # prompt
        print(f"Prompt: {messages[0] if messages else None}")

        try:
            response = client.messages.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )

            # Get text from the content blocks
            response_text = "".join(block.text for block in response.content)

            # Attempt JSON parse
            try:
                data = json.loads(response_text)
                if "instruction" not in data or "output" not in data:
                    raise ValueError("Missing 'instruction' or 'output' fields in JSON.")
            except Exception:
                # fallback if not valid JSON
                data = {
                    "instruction": f"Explain this snippet:\n{snippet_block}",
                    "output": response_text
                }

            # add result
            print(f"First content block: {response.content[0] if response.content else None}")
            fine_tune_data.append(data)

        except Exception as e:
            print(f"Error calling Claude for snippet #{idx}: {e}")

    # 4) Write final dataset
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as out_f:
        json.dump(fine_tune_data, out_f, indent=2, ensure_ascii=False)

    print(f"\nDone! Created '{OUTPUT_JSON_FILE}' with {len(fine_tune_data)} items.")


if __name__ == "__main__":
    main()
