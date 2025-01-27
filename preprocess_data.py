import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(input_file, output_dir):
    # Load the dataset
    df = pd.read_csv(input_file)

    # Identify columns
    twin1_cols = [col for col in df.columns if col.endswith('.1')]
    twin2_cols = [col for col in df.columns if col.endswith('.2')]

    # Normalize traits
    scaler = StandardScaler()
    df[twin1_cols + twin2_cols] = scaler.fit_transform(df[twin1_cols + twin2_cols])

    # Split dataset
    X = df[twin1_cols + twin2_cols]
    y = df['zyg']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Save splits
    for split_name, data in zip(
        ['train', 'val', 'test'],
        [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    ):
        split_df = pd.concat([data[0], data[1]], axis=1)
        split_df.to_csv(f"{output_dir}/{split_name}.csv", index=False)

    print(f"Preprocessed data saved to {output_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess twin zygosity dataset.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save preprocessed data.")
    args = parser.parse_args()

    preprocess_data(args.input_file, args.output_dir)
