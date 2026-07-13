import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(style="whitegrid")


def save_plot(base_dir, filename):
    output_dir = base_dir / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


# -------------------------
# Project root
# -------------------------
base_dir = Path(__file__).resolve().parent.parent

# -------------------------
# Load cleaned dataset (output of Phase 1)
# -------------------------
file_path = base_dir / "data" / "processed" / "clean_dataset.xlsx"
df = pd.read_excel(file_path, engine="openpyxl")

# normalize column names (important for bugs later)
df.columns = df.columns.str.lower().str.strip()

print("Shape:", df.shape)
print(df.head())
df.info()
print(df.isnull().sum())

# -------------------------
# CEFR Distribution
# -------------------------
if "level" in df.columns:
    level_counts = df["level"].value_counts()
    print("\nCEFR level distribution:\n", level_counts)

    plt.figure()
    sns.barplot(x=level_counts.index, y=level_counts.values)
    plt.title("CEFR Level Distribution")
    plt.xlabel("CEFR Level")
    plt.ylabel("Count")
    save_plot(base_dir, "cefr_level_distribution.png")
else:
    print("Column 'level' not found")

# -------------------------
# Category Distribution (main + sub)
# -------------------------
if "main category" in df.columns:
    main_cat_counts = df["main category"].value_counts()
    print("\nMain category distribution:\n", main_cat_counts)

    plt.figure(figsize=(9, 5))
    sns.barplot(y=main_cat_counts.index, x=main_cat_counts.values, orient="h")
    plt.title("Main Category Distribution")
    plt.xlabel("Count")
    plt.ylabel("Main Category")
    save_plot(base_dir, "main_category_distribution.png")
else:
    print("Column 'main category' not found")

if "sub category" in df.columns:
    sub_cat_counts = df["sub category"].value_counts()
    print("\nSub category distribution (top 20):\n", sub_cat_counts.head(20))

    plt.figure(figsize=(9, 8))
    sns.barplot(
        y=sub_cat_counts.head(20).index,
        x=sub_cat_counts.head(20).values,
        orient="h",
    )
    plt.title("Sub Category Distribution (Top 20)")
    plt.xlabel("Count")
    plt.ylabel("Sub Category")
    save_plot(base_dir, "sub_category_distribution.png")
else:
    print("Column 'sub category' not found")

# -------------------------
# Phrase length analysis
# -------------------------
if "phrase" in df.columns:
    phrase_lengths = df["phrase"].astype(str).apply(lambda x: len(x.split()))
    print("\nPhrase length (word count) stats:\n", phrase_lengths.describe())

    plt.figure()
    sns.histplot(phrase_lengths, bins=20)
    plt.title("Phrase Length Distribution (word count)")
    plt.xlabel("Words per phrase")
    save_plot(base_dir, "phrase_length_distribution.png")
else:
    print("Column 'phrase' not found")

# -------------------------
# Data imbalance detection
# -------------------------
if "level" in df.columns:
    imbalance_ratio = level_counts.max() / level_counts.min()
    print(
        f"\nCEFR class imbalance ratio (max/min count): {imbalance_ratio:.2f}"
    )
    if imbalance_ratio > 3:
        print(
            "  Significant class imbalance detected across CEFR levels — "
            "consider this when interpreting Phase 3 classifier metrics "
            "(e.g. prefer macro-F1 over accuracy)."
        )
