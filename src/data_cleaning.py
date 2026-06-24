import pandas as pd

def clean_dataset(file_path, save_path):
    # Load dataset from Excel file
    df = pd.read_excel(file_path)

    # Quick data overview
    print(df.head())
    print(df.shape)

    # Dataset information and missing values check
    print(df.info())
    print(df.isnull().sum())

    # Remove rows with missing values
    df = df.dropna()

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip()

    # Clean text column (lowercase + strip spaces)
    text_col = "phrase"
    df[text_col] = df[text_col].astype(str).str.lower().str.strip()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Save cleaned dataset
    df.to_excel(save_path, index=False)

    print("Saved:", save_path)

    return df