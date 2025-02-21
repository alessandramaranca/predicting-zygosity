import json
import math
import torch
import datetime
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Paths
model_path = "/home/ar0241/scratch/torchtune_models/Llama-3.2-1B-Instruct"
eval_dataset_path = "/home/ar0241/scratch/twins/ptwindat_eval.json"  # updated to match preprocessing output

# Load tokenizer and model.
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load evaluation data.
with open(eval_dataset_path, "r") as f:
    eval_data = json.load(f)

# Prepare lists.
# In the new instruct dataset, "instruction" is empty and "input" contains the twin differences.
prompts = []
targets = []

for example in eval_data:
    # Use the empty instruction; we only use the "input" field.
    input_text = example["input"]
    prompts.append(input_text)
    targets.append(example["output"].strip())

# Define candidate tokens (as strings) for binary classification.
candidates = ["1", "0"]
candidate_token_ids = {
    cand: tokenizer(cand, add_special_tokens=False)["input_ids"] for cand in candidates
}

predictions = []
predicted_probs = []  # Store predicted probability for candidate "1" (for calibration)
p_sums = []           # To track raw probability mass (p(0)+p(1)) per example
additional_details = []  # To store top-token details for each example

# Evaluate each prompt.
for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc="Evaluating"):
    # Tokenize the prompt.
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Get the logits for the next token (for top-3 token check).
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]
    probs_full = torch.softmax(logits, dim=0)
    # Get top 3 tokens from the full vocabulary.
    topk_values, topk_indices = torch.topk(probs_full, k=3)
    top_tokens = [tokenizer.decode([idx]).strip() for idx in topk_indices]

    # Evaluate candidate tokens iteratively (each candidate is expected to be one token).
    candidate_log_probs = {}
    for cand in candidates:
        cand_ids = candidate_token_ids[cand]
        context_ids = input_ids.clone()
        total_log_prob = 0.0
        for token_id in cand_ids:
            with torch.no_grad():
                outputs = model(context_ids)
            logits_step = outputs.logits[0, -1, :]
            log_probs = torch.log_softmax(logits_step, dim=0)
            token_log_prob = log_probs[token_id].item()
            total_log_prob += token_log_prob
            # Append the candidate token for the next step.
            token_tensor = torch.tensor([[token_id]], device=device)
            context_ids = torch.cat([context_ids, token_tensor], dim=1)
        candidate_log_probs[cand] = total_log_prob

    # Convert log probabilities to raw probabilities.
    candidate_probs_raw = {cand: math.exp(candidate_log_probs[cand]) for cand in candidates}
    prob_sum = sum(candidate_probs_raw.values())
    p_sums.append(prob_sum)
    # Normalize to obtain a probability distribution.
    normalized_probs = {cand: candidate_probs_raw[cand] / prob_sum for cand in candidates}

    # Use candidate "1" probability as the predicted positive probability.
    p_positive = normalized_probs["1"]
    predicted_probs.append(p_positive)
    predicted_label = "1" if normalized_probs["1"] >= normalized_probs["0"] else "0"
    predictions.append(predicted_label)

    # Record additional details for this example.
    detail = {
        "p(1)": normalized_probs["1"],
        "p(0)": normalized_probs["0"],
        "p_sum": prob_sum,
        "top_tokens": list(zip(top_tokens, topk_values.cpu().numpy().tolist()))
    }
    additional_details.append(detail)

# Compute classification metrics.
accuracy = accuracy_score(targets, predictions)
precision = precision_score(targets, predictions, pos_label="1", average="binary")
recall = recall_score(targets, predictions, pos_label="1", average="binary")
f1 = f1_score(targets, predictions, pos_label="1", average="binary")
cm = confusion_matrix(targets, predictions, labels=["1", "0"])
report = classification_report(targets, predictions, labels=["1", "0"])

# Compute average raw probability mass for p(0)+p(1) across examples.
avg_p_sum = sum(p_sums) / len(p_sums)

# Compute Brier score.
true_binary = [1 if t == "1" else 0 for t in targets]
brier = brier_score_loss(true_binary, predicted_probs)

# Compute calibration curve (Murphy Curve) using sklearn.
fraction_of_positives, mean_predicted_value = calibration_curve(true_binary, predicted_probs, n_bins=10)

# Plot and save the calibration (Murphy) curve.
plt.figure(figsize=(8, 6))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Calibration curve")
plt.plot([0, 1], [0, 1], "k:", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.title("Calibration Curve (Murphy Curve)")
plt.legend()
murphy_curve_filename = f"/home/ar0241/scratch/twins/{datetime.datetime.now().strftime('%Y-%m-%d')}_murphy_curve.png"
plt.savefig(murphy_curve_filename)
plt.close()

# Save evaluation results to a text file with additional details.
results_filename = f"/home/ar0241/scratch/twins/{datetime.datetime.now().strftime('%Y-%m-%d')}_evalresults.txt"
with open(results_filename, "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Precision (Positive=1): {precision:.4f}\n")
    f.write(f"Recall (Positive=1): {recall:.4f}\n")
    f.write(f"F1 Score (Positive=1): {f1:.4f}\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n")
    f.write("\nClassification Report:\n")
    f.write(report + "\n\n")
    f.write(f"Average p(0)+p(1) across examples: {avg_p_sum:.4f}\n")
    f.write(f"Brier Score: {brier:.4f}\n")
    f.write(f"Murphy Curve saved to: {murphy_curve_filename}\n\n")
    
    # Optionally, record details for the first few examples.
    f.write("Example Details (first 5 examples):\n")
    for i, detail in enumerate(additional_details[:5]):
        f.write(f"Example {i+1}:\n")
        f.write(f"  p(1): {detail['p(1)']:.4f}\n")
        f.write(f"  p(0): {detail['p(0)']:.4f}\n")
        f.write(f"  p(0)+p(1): {detail['p_sum']:.4f}\n")
        f.write("  Top 3 tokens for next token prediction:\n")
        for token, prob in detail["top_tokens"]:
            f.write(f"    Token: {token}, Probability: {prob:.4f}\n")
        f.write("\n")


                                                                                                                    83,1          Bot
