import pandas as pd

df = pd.read_excel("data/processed/clean_dataset.xlsx")
phrases = df["phrase"].tolist()

from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


embeddings = model.encode(
    phrases,
    batch_size=64,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

print(type(embeddings))
print(embeddings.shape)

import numpy as np

np.save(
    "data/embeddings/embeddings.npy",
    embeddings,
)

