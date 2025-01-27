from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import torch

class TwinDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx]),
        }

def load_data(file_path, tokenizer, max_length=512):
    df = pd.read_csv(file_path)
    twin1_cols = [col for col in df.columns if col.endswith('.1')]
    twin2_cols = [col for col in df.columns if col.endswith('.2')]

    def format_example(row):
        traits1 = " ".join([f"Trait{i}:{row[col]:.2f}" for i, col in enumerate(twin1_cols, start=1)])
        traits2 = " ".join([f"Trait{i}:{row[col]:.2f}" for i, col in enumerate(twin2_cols, start=1)])
        return f"Twin1: {traits1} | Twin2: {traits2}"

    df['text'] = df.apply(format_example, axis=1)
    encodings = tokenizer(list(df['text']), truncation=True, padding=True, max_length=max_length)
    labels = list(df['zyg'])

    return TwinDataset(encodings, labels), labels

def evaluate_model(model, tokenizer, data_loader, true_labels, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in data_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, axis=1)
            predictions.extend(preds.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(true_labels, predictions)
    report = classification_report(true_labels, predictions, target_names=["DZ", "MZ"])
    cm = confusion_matrix(true_labels, predictions)

    return acc, report, cm

if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Llama3.2 model on twin zygosity dataset.")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory of the fine-tuned model.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test data CSV file.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    args = parser.parse_args()

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test dataset
    test_dataset, true_labels = load_data(args.test_file, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Evaluate
    acc, report, cm = evaluate_model(model, tokenizer, test_loader, true_labels, device)

    # Print results
    print("\nEvaluation Results")
    print("==================")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

                                                                                                                    83,1          Bot
