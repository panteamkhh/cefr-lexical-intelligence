# Import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Set visualization style
sns.set_theme(style="whitegrid")


# Save plot to the processed data directory
def save_plot(base_dir, filename):
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename

    plt.tight_layout()
    plt.savefig(path, dpi=300)


# Load cleaned dataset
base_dir = Path.cwd().parent
file_path = base_dir / "data" / "clean_dataset.xlsx"

df = pd.read_excel(file_path)

print(df.shape)
print(df.head())


# Dataset overview
df.info()

df.describe(include="all")

print(df.isnull().sum())


# CEFR level distribution
level_counts = df["level"].value_counts()

print(level_counts)

sns.barplot(
    x=level_counts.index,
    y=level_counts.values,
    palette="viridis"
)

plt.title("CEFR Level Distribution")
plt.xlabel("Level")
plt.ylabel("Count")
plt.xticks(rotation=45)

save_plot(base_dir, "cefr_level_distribution.png")

plt.show()


# Main category analysis
main_counts = df["main category"].value_counts()

print(main_counts)

sns.barplot(
    y=main_counts.index,
    x=main_counts.values,
    palette="mako"
)

plt.title("Main Category Distribution")
plt.xlabel("Count")
plt.ylabel("Category")

save_plot(base_dir, "main_categories.png")

plt.show()


# Sub-category analysis
sub_counts = df["sub category"].value_counts()

print(sub_counts.head(20))

sns.set_theme(style="whitegrid")

top20 = sub_counts.head(20)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=top20.values,
    y=top20.index,
    palette="mako"
)

plt.title("Top 20 Sub Categories")
plt.xlabel("Count")
plt.ylabel("Sub Category")

plt.tight_layout()

save_plot(base_dir, "top_20_sub_categories.png")

plt.show()


# Create phrase length feature
df["phrase_length"] = df["phrase"].apply(
    lambda x: len(str(x).split())
)

# Phrase length statistics
print(df["phrase_length"].describe())

# Phrase length distribution
output_dir = base_dir / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(10, 5))

sns.histplot(
    df["phrase_length"],
    bins=20,
    kde=True
)

plt.title("Phrase Length Distribution")
plt.xlabel("Number of Words")
plt.ylabel("Count")

save_plot(base_dir, "phrase_length_distribution.png")

plt.show()


# Relationship analysis: CEFR level vs main category
pivot = pd.crosstab(
    df["level"],
    df["main category"]
)

sns.heatmap(
    pivot,
    cmap="coolwarm",
    annot=False
)

plt.title("CEFR Level vs Main Category")

save_plot(base_dir, "cefr_vs_main_category.png")

plt.show()


# Create phrase length buckets
df["length_bucket"] = pd.cut(
    df["phrase_length"],
    bins=[0, 1, 2, 3, 10],
    labels=["1", "2", "3", "4+"]
)

# Relationship analysis: CEFR level vs phrase length
pivot2 = pd.crosstab(
    df["level"],
    df["length_bucket"]
)

sns.heatmap(
    pivot2,
    annot=True,
    cmap="YlGnBu",
    fmt="g"
)

plt.title("CEFR Level vs Phrase Length")

save_plot(base_dir, "cefr_vs_phrase_length.png")

plt.show()