from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments 
import pandas as pd
import torch

# Set Hugging Face access token
import os
os.environ["HF_HOME"] = "/path/to/cache"  # Optional: To set a custom cache location
os.environ["HF_TOKEN"] = "add_your_token_here"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=os.environ["HF_TOKEN"])
tokenizer.pad_token = tokenizer.eos_token 
model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B", use_auth_token=os.environ["HF_TOKEN"], num_labels=2)


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
        label = "DZ" if row['zyg'] == 0 else "MZ"
        return f"Twin1: {traits1} | Twin2: {traits2} | Zygosity: {label}"
    df['text'] = df.apply(format_example, axis=1) 
    encodings = tokenizer(list(df['text']), truncation=True, padding=True, max_length=max_length) 
    labels = list(df['zyg']) 
 
    return TwinDataset(encodings, labels) 
 
if __name__ == "__main__": 
    import argparse 
 
    parser = argparse.ArgumentParser(description="Fine-tune Llama3.2 for twin zygosity prediction.") 
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data.") 
    parser.add_argument("--val_file", type=str, required=True, help="Path to the validation data.") 
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model.") 
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.") 
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.") 
    args = parser.parse_args() 
 
    # Load tokenizer and model 
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B") 
    tokenizer.pad_token = tokenizer.eos_token  # Use eos_token as pad_token 
    model = AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B", num_labels=2) 
 
    # Load datasets 
    train_dataset = load_data(args.train_file, tokenizer) 
    val_dataset = load_data(args.val_file, tokenizer) 
 
 
    # Training arguments 
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    # Trainer 
    trainer = Trainer( 
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        tokenizer=tokenizer, 
    ) 
 
    # Train 
    trainer.train() 
 
    # Save model 
    model.save_pretrained(args.output_dir) 
    tokenizer.save_pretrained(args.output_dir) 
    print(f"Model saved to {args.output_dir}") 
