import os

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    """
    Prepare dataset splits (train/valid/test) in an 8:1:1 ratio.

    This is a CLI equivalent of the dataset preparation logic in app.py, so the
    project can be used from the terminal without the Streamlit UI.
    """
    data_path = os.path.join("dataset", "data.csv")

    if not os.path.exists(data_path):
        raise FileNotFoundError(
            "dataset/data.csv not found. "
            "Run download_dataset.sh or use the Streamlit app to download it first."
        )

    print("Loading raw dataset...")
    raw = pd.read_csv(data_path)
    orig_len = len(raw)
    print(f"Original rows: {orig_len}")

    # Remove exact duplicates
    dupes = raw.duplicated().sum()
    if dupes > 0:
        raw = raw.drop_duplicates()
        print(f"Removed {dupes} duplicate rows.")

    target = "disease_diagnosis"
    if target not in raw.columns:
        raise KeyError(f"Target column '{target}' not found in dataset.")

    # Stratified 8:1:1 split preserves disease class ratios
    print("Performing stratified 8:1:1 split (train/valid/test)...")
    train_df, temp_df = train_test_split(
        raw, test_size=0.2, random_state=42, stratify=raw[target]
    )
    valid_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df[target]
    )

    os.makedirs("dataset", exist_ok=True)
    train_path = os.path.join("dataset", "train.csv")
    valid_path = os.path.join("dataset", "valid.csv")
    test_path = os.path.join("dataset", "test.csv")

    train_df.to_csv(train_path, index=False)
    valid_df.to_csv(valid_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Dataset prepared successfully.")
    print(f"After dedup (if any): {len(raw)} rows")
    print(f"Train: {len(train_df)} rows (~80%) -> {train_path}")
    print(f"Valid: {len(valid_df)} rows (~10%) -> {valid_path}")
    print(f"Test:  {len(test_df)} rows (~10%) -> {test_path}")


if __name__ == "__main__":
    main()

