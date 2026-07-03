import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

sns.set_theme(style="whitegrid")


def save_plot(base_dir, filename):
    output_dir = base_dir / "data" / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# -------------------------
# Project root
# -------------------------
base_dir = Path(__file__).resolve().parents[1]

# -------------------------
# Load Excel
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

    plt.figure()
    sns.barplot(x=level_counts.index, y=level_counts.values)
    plt.title("CEFR Level Distribution")

    save_plot(base_dir, "cefr_level_distribution.png")
    plt.show()
else:
    print("Column 'level' not found")

# -------------------------
# Phrase length analysis
# -------------------------
if "phrase" in df.columns:
    phrase_lengths = df["phrase"].astype(str).apply(lambda x: len(x.split()))
    print(phrase_lengths.describe())
else:
    print("Column 'phrase' not found")