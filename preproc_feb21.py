import pandas as pd
import json
import numpy as np

# Load the CSV file.
file_path = "/home/ar0241/scratch/twins/twindat_sim_100k_24.csv"
df = pd.read_csv(file_path)

# Identify traits (assumes trait columns end with '.1' and '.2').
traits = sorted(set(col[:-2] for col in df.columns if col.endswith('.1')))

# For each trait, compute the absolute difference between the twins.
for trait in traits:
    col1 = trait + '.1'
    col2 = trait + '.2'
    # Compute the absolute difference between the raw values.
    df[f"diff_{trait}"] = np.abs(df[col1] - df[col2])
  
# Create the instruct-format data.
def row_to_instruct(row):
    input_lines = []
    for trait in traits:
        diff_val = row[f"diff_{trait}"]
        input_lines.append(f"{trait} difference: {diff_val:.2f}")
    # Join each trait difference with commas.
    input_text = ", ".join(input_lines)
    # Output is "1" if monozygotic (zyg==1), else "0".
    output_text = "1" if row['zyg'] == 1 else "0"
    return {"instruction": "", "input": input_text, "output": output_text}

instruct_data = df.apply(row_to_instruct, axis=1).tolist()

# Shuffle the data (ensuring reproducibility).
np.random.seed(42)
np.random.shuffle(instruct_data)

# Split dataset: 80% train, 10% validation, 10% evaluation.
total_samples = len(instruct_data)
train_split = int(total_samples * 0.8)
val_split = int(total_samples * 0.9)

train_data = instruct_data[:train_split]
val_data = instruct_data[train_split:val_split]
eval_data = instruct_data[val_split:]

# Save the split datasets as JSON files.
with open("/home/ar0241/scratch/twins/ptwindat_train.json", "w") as f:
    json.dump(train_data, f, indent=4)

with open("/home/ar0241/scratch/twins/ptwindat_val.json", "w") as f:
    json.dump(val_data, f, indent=4)

with open("/home/ar0241/scratch/twins/ptwindat_eval.json", "w") as f:
    json.dump(eval_data, f, indent=4)

print("Saved train (80%), validation (10%), and evaluation (10%) datasets.")

