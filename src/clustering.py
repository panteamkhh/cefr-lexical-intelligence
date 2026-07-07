# ============================================================
# Imports
# ============================================================

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap

from sklearn.decomposition import PCA
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)

# ============================================================
# Define Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EMBEDDING_DIR = DATA_DIR / "embeddings"
CLUSTERING_DIR = DATA_DIR / "clustering"
FIGURE_DIR = DATA_DIR / "figures"

# ============================================================
# Load Embedding Artifacts
# ============================================================

embeddings = np.load(
    EMBEDDING_DIR / "embeddings.npy"
)

metadata = pd.read_csv(
    EMBEDDING_DIR / "metadata.csv"
)

with open(
    EMBEDDING_DIR / "embedding_config.json",
    "r",
    encoding="utf-8"
) as f:
    config = json.load(f)

# ============================================================
# Consistency Check
# ============================================================

assert embeddings.shape[0] == len(metadata)
assert embeddings.shape[1] == config["dimension"]
assert len(metadata) == config["num_samples"]

print("✅ Embedding artifacts are consistent.")

# ============================================================
# Create Output Directories
# ============================================================

CLUSTERING_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# Task 2 - PCA Projection
# ============================================================

pca = PCA(
    n_components=2,
    random_state=42
)

pca_embeddings = pca.fit_transform(
    embeddings
)

print("PCA shape:", pca_embeddings.shape)
print(
    "Explained variance:",
    pca.explained_variance_ratio_.sum()
)

# ============================================================
# PCA Visualization
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    pca_embeddings[:, 0],
    pca_embeddings[:, 1],
    s=10
)

plt.title("PCA Projection of Embedding Space")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "pca_projection.png",
    dpi=300
)

plt.show()

# ============================================================
# UMAP Projection
# ============================================================

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

umap_embeddings = umap_model.fit_transform(
    embeddings
)

print(
    "UMAP shape:",
    umap_embeddings.shape
)

# ============================================================
# UMAP Visualization
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    umap_embeddings[:, 0],
    umap_embeddings[:, 1],
    s=10
)

plt.title("UMAP Projection of Embedding Space")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "umap_projection.png",
    dpi=300
)

plt.show()

# ============================================================
# Save Projection Data
# ============================================================

projection_df = pd.DataFrame(
    {
        "sample_id": np.arange(len(embeddings)),
        "pca_1": pca_embeddings[:, 0],
        "pca_2": pca_embeddings[:, 1],
        "umap_1": umap_embeddings[:, 0],
        "umap_2": umap_embeddings[:, 1],
    }
)

projection_df.to_csv(
    CLUSTERING_DIR / "embedding_projections.csv",
    index=False
)

print("✅ Projection files saved.")

# ============================================================
# Task 3 - KMeans Clustering
# ============================================================

k_values = range(2, 11)

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(
        embeddings
    )

    pd.DataFrame(
        {
            "sample_id": np.arange(len(labels)),
            "cluster": labels
        }
    ).to_csv(
        CLUSTERING_DIR / f"kmeans_k{k}_labels.csv",
        index=False
    )

print("✅ KMeans clustering completed.")

# ============================================================
# Task 3 - DBSCAN Clustering
# ============================================================

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5,
    metric="cosine"
)

dbscan_labels = dbscan.fit_predict(
    embeddings
)

pd.DataFrame(
    {
        "sample_id": np.arange(len(dbscan_labels)),
        "cluster": dbscan_labels
    }
).to_csv(
    CLUSTERING_DIR / "dbscan_labels.csv",
    index=False
)

print("✅ DBSCAN clustering completed.")

# ============================================================
# Task 3 - Hierarchical Clustering
# ============================================================

hierarchical = AgglomerativeClustering(
    n_clusters=5,
    metric="cosine",
    linkage="average"
)

hierarchical_labels = hierarchical.fit_predict(
    embeddings
)

pd.DataFrame(
    {
        "sample_id": np.arange(len(hierarchical_labels)),
        "cluster": hierarchical_labels
    }
).to_csv(
    CLUSTERING_DIR / "hierarchical_labels.csv",
    index=False
)

print("✅ Hierarchical clustering completed.")

