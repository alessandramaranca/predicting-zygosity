import json
import re
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# Define paths to your model and evaluation dataset.
model_path = "/home/ar0241/scratch/torchtune_models/Llama-3.2-1B-Instruct"
eval_dataset_path = "/home/ar0241/scratch/twins/twindat_eval.json"

# Load the tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained(model_path)
# Set the pad token if it's not defined.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path)

# Set up device: use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # Set to evaluation mode.

# Load the evaluation data.
with open(eval_dataset_path, "r") as f:
    eval_data = json.load(f)

# Prepare the list of prompts and targets.
prompts = []
targets = []

for example in eval_data:
    instruction = example["instruction"]
    input_text = example["input"]
    target = example["output"].strip()  # Expected: "Monozygotic" or "Dizygotic"
    prompt = (
        f"Instruction: {instruction}\n"
        f"Input: {input_text}\n"
        "Please answer with only one word: either 'Monozygotic' or 'Dizygotic'.\n"
        "Response:"
    )
    prompts.append(prompt)
    targets.append(target)
# Define batch size (adjust based on your GPU memory).
batch_size = 16
num_batches = math.ceil(len(prompts) / batch_size)

# These lists will hold the final predictions and corresponding true labels.
predictions = []
actuals = []

# Iterate over batches with a progress bar.
for i in tqdm(range(num_batches), desc="Evaluating"):
    batch_prompts = prompts[i * batch_size : (i + 1) * batch_size]
    batch_targets = targets[i * batch_size : (i + 1) * batch_size]

    # Tokenize batch with padding/truncation.
    batch_inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
    # Move inputs to the GPU.
    batch_inputs = {k: v.to(device) for k, v in batch_inputs.items()}

    # Generate outputs for the batch.
    with torch.no_grad():
        batch_outputs = model.generate(
            **batch_inputs,
            max_new_tokens=20,
            do_sample=False  # Greedy decoding.
        )

    # Decode the outputs.
    decoded_texts = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)

    # Process each output in the batch.
    for pred_text, target in zip(decoded_texts, batch_targets):
        # Use regex to extract only "Monozygotic" or "Dizygotic" (case-insensitive).
        match = re.search(r'\b(monozygotic|dizygotic)\b', pred_text, flags=re.IGNORECASE)
        if match:
            predicted_label = match.group(1).capitalize()
        else:
            predicted_label = ""

        predictions.append(predicted_label)
        actuals.append(target)

# Compute traditional binary classification metrics.
accuracy = accuracy_score(actuals, predictions)
# In these metrics, we treat "Monozygotic" as the positive class.
precision = precision_score(actuals, predictions, pos_label="Monozygotic", average="binary")
recall = recall_score(actuals, predictions, pos_label="Monozygotic", average="binary")
f1 = f1_score(actuals, predictions, pos_label="Monozygotic", average="binary")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision (Monozygotic as positive): {precision:.4f}")
print(f"Recall (Monozygotic as positive): {recall:.4f}")
print(f"F1 Score (Monozygotic as positive): {f1:.4f}")

# Compute and print the confusion matrix.
cm = confusion_matrix(actuals, predictions, labels=["Monozygotic", "Dizygotic"])
print("Confusion Matrix:")
print(cm)

# Print a detailed classification report.
report = classification_report(actuals, predictions, labels=["Monozygotic", "Dizygotic"])
print("\nClassification Report:")
print(report)
