import pandas as pd
from pathlib import Path


def clean_dataset(file_path, save_path):
    file_path = Path(file_path)

    # -------------------------
    # Load dataset 
    # -------------------------
    df = pd.read_excel(file_path, engine="openpyxl")

    # -------------------------
    # Standardize columns
    # -------------------------
    df.columns = df.columns.str.lower().str.strip()

    print("Shape:", df.shape)
    print(df.head())
    df.info()
    print(df.isnull().sum())

    # -------------------------
    # Validate required column
    # -------------------------
    required_cols = ["phrase"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    # -------------------------
    # Clean data
    # -------------------------
    df = df.dropna(subset=["phrase"])

    df["phrase"] = (
        df["phrase"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    df = df.drop_duplicates()

    # -------------------------
    # Save output
    # -------------------------
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("Saved:", save_path)

    return df