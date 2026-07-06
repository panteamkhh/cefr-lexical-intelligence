# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Load Dataset
# =========================
df = pd.read_excel("data/processed/clean_dataset.xlsx")
phrases = df["phrase"].tolist()


# =========================
# Load Embedding Model
# =========================
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# =========================
# Generate Embeddings
# =========================
embeddings = model.encode(
    phrases,
    batch_size=64,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)


# =========================
# Validate Embeddings
# =========================
print("Shape:", embeddings.shape)
print("NaN exists:", np.isnan(embeddings).any())
print("Inf exists:", np.isinf(embeddings).any())

norms = np.linalg.norm(embeddings, axis=1)
print("Min norm:", norms.min())
print("Max norm:", norms.max())


# =========================
# Save Embeddings
# =========================
np.save("data/embeddings/embeddings.npy", embeddings)


# =========================
# Build Metadata
# =========================
metadata = df.copy()
metadata.insert(0, "id", range(1, len(metadata) + 1))

metadata = metadata.drop(columns=["#"])

metadata.rename(
    columns={
        "main category": "main_category",
        "sub category": "sub_category",
    },
    inplace=True,
)

metadata = metadata[
    [
        "id",
        "phrase",
        "level",
        "main_category",
        "sub_category",
    ]
]

metadata.to_csv(
    "data/embeddings/metadata.csv",
    index=False,
    encoding="utf-8-sig",
)


# =========================
# Save Config
# =========================
config = {
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "dimension": 384,
    "normalize_embeddings": True,
    "dataset": "clean_dataset.xlsx",
    "num_samples": len(df)
}

with open(
    "data/embeddings/embedding_config.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(config, f, indent=4, ensure_ascii=False)


# =========================
# Semantic Validation 
# =========================
def sim(a, b):
    return cosine_similarity([a], [b])[0][0]


teacher = embeddings[0]
student = embeddings[1]
school = embeddings[2]
car = embeddings[10]

print("teacher vs student:", sim(teacher, student))
print("teacher vs school:", sim(teacher, school))
print("teacher vs car:", sim(teacher, car))

economy = embeddings[3]
emotion = embeddings[4]
vehicle = embeddings[10]

print("economy vs emotion:", sim(economy, emotion))
print("economy vs vehicle:", sim(economy, vehicle))
print("emotion vs vehicle:", sim(emotion, vehicle))