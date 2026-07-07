# Prepare the Embedding Space
# Imports

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering
)
from sklearn.metrics import pairwise_distances


# Define Project Root
# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDING_DIR = DATA_DIR / "embeddings"
CLUSTERING_DIR = DATA_DIR / "clustering"
FIGURE_DIR = DATA_DIR / "figures"

# Load Files
embeddings = np.load( EMBEDDING_DIR / "embeddings.npy")

metadata = pd.read_csv(EMBEDDING_DIR / "metadata.csv")

with open(
    EMBEDDING_DIR / "embedding_config.json",
    "r",
    encoding="utf-8"
) as f:
    config = json.load(f)

# Consistency Check
assert embeddings.shape[0] == len(metadata)
assert embeddings.shape[1] == config["dimension"]
assert len(metadata) == config["num_samples"]

print("✅ Embedding artifacts are consistent.")

# Create Output Folder
CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# PCA Projection
# =========================

pca = PCA(
    n_components=2,
    random_state=42
)

pca_embeddings = pca.fit_transform(embeddings)


print("PCA shape:", pca_embeddings.shape)

print(
    "Explained variance:",
    pca.explained_variance_ratio_.sum()
)

# =========================
# PCA Visualization
# =========================

plt.figure(figsize=(8,6))

plt.scatter(
    pca_embeddings[:,0],
    pca_embeddings[:,1],
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

# =========================
# UMAP Projection
# =========================

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

# =========================
# UMAP Visualization
# =========================

plt.figure(figsize=(8,6))


plt.scatter(
    umap_embeddings[:,0],
    umap_embeddings[:,1],
    s=10
)


plt.title(
    "UMAP Projection of Embedding Space"
)

plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")


plt.tight_layout()


plt.savefig(
    FIGURE_DIR / "umap_projection.png",
    dpi=300
)


plt.show()

# =========================
# Save Projections
# =========================

projection_df = pd.DataFrame(
    {
        "pca_1": pca_embeddings[:,0],
        "pca_2": pca_embeddings[:,1],
        "umap_1": umap_embeddings[:,0],
        "umap_2": umap_embeddings[:,1],
    }
)


projection_df.to_csv(
    CLUSTERING_DIR / "embedding_projections.csv",
    index=False
)


print("✅ Projection files saved.")




