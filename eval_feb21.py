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
    roc_auc_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import pandas as pd

# Paths
model_path = "/home/ar0241/scratch/torchtune_models/Llama-3.2-1B-Instruct"
eval_dataset_path = "/home/ar0241/scratch/twins/ptwindat_eval.json"

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
prompts = []
targets = []
for example in eval_data:
    input_text = example["input"]
    prompts.append(input_text)
    targets.append(example["output"].strip())

# Define candidate tokens for binary classification.
candidates = ["1", "0"]
# For single-token candidates, take the first token id.
candidate_token_ids = {cand: tokenizer(cand, add_special_tokens=False)["input_ids"][0] for cand in candidates}

predictions = []
predicted_probs = []  # Predicted probability for candidate "1"
additional_details = []  # To store candidate probabilities per example
losses = []  # Negative log-likelihood loss per example

# Evaluate each prompt.
for prompt, target in tqdm(zip(prompts, targets), total=len(prompts), desc="Evaluating"):
    # Tokenize the prompt.
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Get logits for the next token.
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits[0, -1, :]

    # Compute full softmax probabilities.
    softmax_probs = torch.softmax(logits, dim=0)

    # Get candidate probabilities directly.
    cand1_prob = softmax_probs[candidate_token_ids["1"]].item()
    cand0_prob = softmax_probs[candidate_token_ids["0"]].item()
    candidate_sum = cand1_prob + cand0_prob
    # Normalize over candidates so that they sum to 1.
    normalized_probs = {
        "1": cand1_prob / candidate_sum,
        "0": cand0_prob / candidate_sum
    }
    # For reporting, set p_sum to 1 (since we normalized over the two candidates).
    p_sum = 1.0

    predicted_probs.append(normalized_probs["1"])
    predicted_label = "1" if normalized_probs["1"] >= normalized_probs["0"] else "0"
    predictions.append(predicted_label)

    # Compute loss: negative log likelihood for the correct candidate.
    loss = -math.log(normalized_probs[target])
    losses.append(loss)

    # Prepare sorted candidate probabilities.
    sorted_cands = sorted(normalized_probs.items(), key=lambda x: x[1], reverse=True)
    top_candidate_probs = [prob for _, prob in sorted_cands]
    # If fewer than 3 candidates, pad with N/A.
    while len(top_candidate_probs) < 3:
        top_candidate_probs.append("N/A")

    detail = {
        "p(1)": normalized_probs["1"],
        "p(0)": normalized_probs["0"],
        "p_sum": p_sum,
        "top_candidate_probs": top_candidate_probs
    }
    additional_details.append(detail)

# Compute classification metrics.
accuracy = accuracy_score(targets, predictions)
precision = precision_score(targets, predictions, pos_label="1", average="binary")
recall = recall_score(targets, predictions, pos_label="1", average="binary")
f1 = f1_score(targets, predictions, pos_label="1", average="binary")
cm = confusion_matrix(targets, predictions, labels=["1", "0"])
report = classification_report(targets, predictions, labels=["1", "0"])

# Compute Brier score.
true_binary = [1 if t == "1" else 0 for t in targets]
brier = brier_score_loss(true_binary, predicted_probs)

# Compute AUC score.
auc = roc_auc_score(true_binary, predicted_probs)

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

# Compute and save evaluation loss per batch to CSV.
batch_size = 8
batch_loss_data = []
for i in range(0, len(losses), batch_size):
    batch_losses = losses[i:i+batch_size]
    avg_loss = sum(batch_losses) / len(batch_losses)
    batch_loss_data.append({"batch": i // batch_size, "avg_loss": avg_loss})
loss_csv_filename = f"/home/ar0241/scratch/twins/{datetime.datetime.now().strftime('%Y-%m-%d')}_eval_loss.csv"
df_loss = pd.DataFrame(batch_loss_data)
df_loss.to_csv(loss_csv_filename, index=False)

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
    f.write(f"Brier Score: {brier:.4f}\n")
    f.write(f"AUC Score: {auc:.4f}\n")
    f.write(f"Murphy Curve saved to: {murphy_curve_filename}\n")
    f.write(f"Evaluation Loss CSV saved to: {loss_csv_filename}\n\n")

    # Record details for the first 5 examples.
    f.write("Example Details (first 5 examples):\n")
    for i, detail in enumerate(additional_details[:5]):
        f.write(f"Example {i+1}:\n")
        f.write(f"  p(1): {detail['p(1)']:.4f}\n")
        f.write(f"  p(0): {detail['p(0)']:.4f}\n")
        f.write(f"  p(0)+p(1): {detail['p_sum']:.4f}\n")
        f.write("  Top candidate probabilities (sorted):\n")
        for j, prob in enumerate(detail["top_candidate_probs"]):
            f.write(f"    Choice {j+1}: {prob if isinstance(prob, str) else f'{prob:.4f}'}\n")
        f.write("\n")



                                                                                                                    83,1          Bot
