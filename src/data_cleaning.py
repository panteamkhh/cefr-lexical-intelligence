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
    # Drop incidental index/counter columns (e.g. "#", "unnamed: 0")
    # -------------------------
    index_like_cols = [
        c for c in df.columns if c == "#" or c.startswith("unnamed")
    ]
    if index_like_cols:
        print("Dropping index-like columns before dedup:", index_like_cols)
        df = df.drop(columns=index_like_cols)

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

    # -------------------------
    # Deduplicate on the MEANINGFUL content columns only
    # -------------------------
    dedup_subset = [
        c for c in ["phrase", "level", "main category", "sub category"]
        if c in df.columns
    ]
    before = len(df)
    df = df.drop_duplicates(subset=dedup_subset)
    print(f"Dropped {before - len(df)} exact duplicate rows (subset={dedup_subset})")

    df = df.reset_index(drop=True)

    # -------------------------
    # Flag (do NOT silently drop) phrases with conflicting CEFR levels
    # -------------------------
    if "level" in df.columns:
        level_counts = df.groupby("phrase")["level"].nunique()
        conflicting = level_counts[level_counts > 1]
        if len(conflicting) > 0:
            print(
                f"⚠️  {len(conflicting)} phrases have conflicting CEFR levels "
                f"({df['phrase'].isin(conflicting.index).sum()} rows involved). "
                "Kept as-is; review manually if needed."
            )

    # -------------------------
    # Save output
    # -------------------------
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # FIX
    if save_path.suffix.lower() in (".xlsx", ".xls"):
        df.to_excel(save_path, index=False)
    else:
        df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("Saved:", save_path)

    return df


if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parent.parent

    clean_dataset(
        file_path=BASE_DIR / "data" / "raw" / "toefl_vocabulary_cleaned_categorized.xlsx",
        save_path=BASE_DIR / "data" / "processed" / "clean_dataset.xlsx",
    )
