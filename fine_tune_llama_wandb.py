import pandas as pd
import torch
import torchtune as tt
import wandb
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Initialize WandB
wandb.init(project="zygosity-twin-llama")

def preprocess_data(input_file, output_dir):
    df = pd.read_csv(input_file)
    twin1_cols = [col for col in df.columns if col.endswith('.1')]
    twin2_cols = [col for col in df.columns if col.endswith('.2')]
    
    scaler = StandardScaler()
    df[twin1_cols + twin2_cols] = scaler.fit_transform(df[twin1_cols + twin2_cols])
    
    X = df[twin1_cols + twin2_cols]
    y = df['zyg']
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    for split_name, data in zip(['train', 'val', 'test'], [(X_train, y_train), (X_val, y_val), (X_test, y_test)]):
        split_df = pd.concat([data[0], data[1]], axis=1)
        split_df.to_csv(f"{output_dir}/{split_name}.csv", index=False)
    
    print(f"Preprocessed data saved to {output_dir}")

class TwinDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}, torch.tensor(self.labels[idx])

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
    labels = df['zyg'].tolist()
    
    return TwinDataset(encodings, labels)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 with Torchtune and WandB.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_dataset = load_data(args.train_file, tokenizer)
    val_dataset = load_data(args.val_file, tokenizer)
    
    model = tt.models.LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B", num_labels=2)
    
    trainer = tt.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        output_dir=args.output_dir,
        wandb_project="zygosity-twin-llama",
        eval_steps=500,
        save_steps=500,
        save_total_limit=2,
    )
    
    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    wandb.finish()
    print(f"Model saved to {args.output_dir}")
