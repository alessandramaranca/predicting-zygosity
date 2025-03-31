import json
import sys
from transformers import AutoTokenizer

# Config
INPUT_FILE_PREFIX = "/home/niznik/scratch/zyg_in/nd/ptwindat"
OUTPUT_FILE_PREFIX = "/home/niznik/scratch/zyg_in/nd/ptwindat" 

for split_name in ['train', 'eval', 'val']:
    with open(f"{INPUT_FILE_PREFIX}_{split_name}.json") as f:
        original_json = json.load(f)

    prompts = []

    for entry in original_json:
        input_text = entry["input"].strip()
        label = entry["output"].strip()

        prompt = f"{input_text}, {label}"
        prompts.append(prompt)

    with open(f"{OUTPUT_FILE_PREFIX}_{split_name}_single.jsonl", "w") as out:
        for p in prompts:
            out.write(json.dumps({"text": p}) + "\n")