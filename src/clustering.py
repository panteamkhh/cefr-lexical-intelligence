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

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
from sklearn.metrics.pairwise import cosine_distances

# ============================================================
# Define Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
EMBEDDING_DIR = DATA_DIR / "embeddings"
CLUSTERING_DIR = DATA_DIR / "clustering"
FIGURE_DIR = DATA_DIR / "figures"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "clustering"
INTERMEDIATE_DIR = EXPERIMENTS_DIR / "intermediate_results"
MODEL_SEARCH_DIR = EXPERIMENTS_DIR / "model_search_results"
EXPERIMENTAL_PLOTS_DIR = EXPERIMENTS_DIR / "experimental_plots"

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
assert len(embeddings) == len(metadata)
assert embeddings.shape[0] == len(metadata)
assert embeddings.shape[1] == config["dimension"]
assert len(metadata) == config["num_samples"]


_dup_cols = ["phrase", "level", "main_category", "sub_category"]
_n_dupes = metadata.duplicated(subset=_dup_cols).sum()
assert _n_dupes == 0, (
    f"Found {_n_dupes} exact duplicate rows in metadata.csv - "
    "re-run Phase 1 (data_cleaning.py) first."
)

print(" Embedding artifacts are consistent.")

# ============================================================
# Create Output Directories
# ============================================================

CLUSTERING_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# Experiment dirs are for intermediate/search artifacts only.
INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_SEARCH_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTAL_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# A stable sample_id used consistently across every artifact in this phase.
sample_ids = np.arange(len(embeddings))

# ============================================================
# Task 1 - PCA Projection
# ============================================================

pca = PCA(
    n_components=2,
    random_state=42
)

pca_embeddings = pca.fit_transform(embeddings)

print("PCA shape:", pca_embeddings.shape)
print("Explained variance:", pca.explained_variance_ratio_.sum())

plt.figure(figsize=(8, 6))
plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], s=10)
plt.title("PCA Projection of Embedding Space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "pca_projection.png", dpi=300)
plt.close()

# ============================================================
# Task 1 - UMAP Projection (cosine metric, semantic vectors)
# ============================================================

umap_model = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.1,
    metric="cosine",
    random_state=42
)

umap_embeddings = umap_model.fit_transform(embeddings)

print("UMAP shape:", umap_embeddings.shape)

plt.figure(figsize=(8, 6))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=10)
plt.title("UMAP Projection of Embedding Space")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "umap_projection.png", dpi=300)
plt.close()

# ============================================================
# Save Projection Data (reused later - never recomputed)
# ============================================================

projection_df = pd.DataFrame(
    {
        "sample_id": sample_ids,
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

print(" Projection files saved.")

# ============================================================
# Task 2 - KMeans Clustering (k = 2..10)
# ============================================================


k_values = range(2, 11)
kmeans_models = {}

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )
    labels = kmeans.fit_predict(embeddings)
    kmeans_models[k] = kmeans

    pd.DataFrame(
        {
            "sample_id": sample_ids,
            "cluster": labels
        }
    ).to_csv(
        MODEL_SEARCH_DIR / f"kmeans_k{k}_labels.csv",
        index=False
    )

print(" KMeans clustering completed (search artifacts in experiments/).")

# ============================================================
# Task 2 - DBSCAN Clustering
# ============================================================

dbscan = DBSCAN(
    eps=0.5,
    min_samples=5,
    metric="cosine"
)

dbscan_labels = dbscan.fit_predict(embeddings)

pd.DataFrame(
    {
        "sample_id": sample_ids,
        "cluster": dbscan_labels
    }
).to_csv(
    CLUSTERING_DIR / "dbscan_labels.csv",
    index=False
)

print(" DBSCAN clustering completed.")

# ============================================================
# Task 2 - Hierarchical Clustering
# ============================================================

hierarchical = AgglomerativeClustering(
    n_clusters=5,
    metric="cosine",
    linkage="average"
)

hierarchical_labels = hierarchical.fit_predict(embeddings)

pd.DataFrame(
    {
        "sample_id": sample_ids,
        "cluster": hierarchical_labels
    }
).to_csv(
    CLUSTERING_DIR / "hierarchical_labels.csv",
    index=False
)

print(" Hierarchical clustering completed.")

# ============================================================
# Task 3 - Evaluate Clustering Models
# ============================================================

metrics_results = []


def evaluate_clustering(algorithm, labels, k=None):
    """
    Evaluate clustering quality using Silhouette, Davies-Bouldin and
    Calinski-Harabasz. Noise points (-1, e.g. from DBSCAN) are excluded.
    """
    labels = np.asarray(labels)
    unique_clusters = set(labels.tolist())
    valid_clusters = unique_clusters - {-1}

    if len(valid_clusters) < 2:
        print(f" {algorithm} (k={k}) skipped: fewer than 2 valid clusters")
        return

    if -1 in unique_clusters:
        mask = labels != -1
        embeddings_eval = embeddings[mask]
        labels_eval = labels[mask]
    else:
        embeddings_eval = embeddings
        labels_eval = labels

    silhouette = silhouette_score(
        embeddings_eval, labels_eval, metric="cosine"
    )
    davies_bouldin = davies_bouldin_score(embeddings_eval, labels_eval)
    calinski_harabasz = calinski_harabasz_score(embeddings_eval, labels_eval)

    metrics_results.append(
        {
            "algorithm": algorithm,
            "k": k,
            "silhouette": silhouette,
            "davies_bouldin": davies_bouldin,
            "calinski_harabasz": calinski_harabasz,
            "num_clusters": len(valid_clusters)
        }
    )


for k, model in kmeans_models.items():
    evaluate_clustering(algorithm="KMeans", labels=model.labels_, k=k)

evaluate_clustering(algorithm="DBSCAN", labels=dbscan_labels)
evaluate_clustering(algorithm="Hierarchical", labels=hierarchical_labels, k=5)

# ============================================================
# Comparison Table
# ============================================================

metrics_df = pd.DataFrame(metrics_results)
metrics_df = metrics_df.sort_values(by="silhouette", ascending=False).reset_index(drop=True)

metrics_df.to_csv(
    CLUSTERING_DIR / "comparison_table.csv",
    index=False
)

print(metrics_df)

# ============================================================
# Task 2 (final step) - Always persist the BEST KMeans result
# ============================================================
# This must happen regardless of whether KMeans is the overall winner:
# the spec requires a single best-of-KMeans artifact independent of the
# cross-algorithm comparison.

kmeans_only = metrics_df[metrics_df["algorithm"] == "KMeans"]
best_kmeans_row = kmeans_only.sort_values("silhouette", ascending=False).iloc[0]
best_kmeans_k = int(best_kmeans_row["k"])
best_kmeans_labels = kmeans_models[best_kmeans_k].labels_

pd.DataFrame(
    {
        "sample_id": sample_ids,
        "cluster": best_kmeans_labels
    }
).to_csv(
    CLUSTERING_DIR / "kmeans_labels.csv",
    index=False
)

print(f" Best KMeans result saved (k={best_kmeans_k}).")

# ============================================================
# Task 4 - Select Best Model Overall (highest silhouette)
# ============================================================

best_result = metrics_df.iloc[0]

best_model_entry = {
    "algorithm": best_result["algorithm"],
    "k": (
        int(best_result["k"]) if not pd.isna(best_result["k"]) else None
    ),
    "silhouette": float(best_result["silhouette"]),
    "davies_bouldin": float(best_result["davies_bouldin"]),
    "calinski_harabasz": float(best_result["calinski_harabasz"])
}

all_results_entries = [
    {
        "algorithm": row["algorithm"],
        "k": (int(row["k"]) if not pd.isna(row["k"]) else None),
        "silhouette": float(row["silhouette"]),
        "davies_bouldin": float(row["davies_bouldin"]),
        "calinski_harabasz": float(row["calinski_harabasz"]),
        "num_clusters": int(row["num_clusters"])
    }
    for _, row in metrics_df.iterrows()
]

cluster_metrics = {
    "best_model": best_model_entry,
    "all_results": all_results_entries,
    "selection_criterion": "highest_silhouette_score"
}

with open(CLUSTERING_DIR / "cluster_metrics.json", "w", encoding="utf-8") as f:
    json.dump(cluster_metrics, f, indent=4)

# ============================================================
# Save Clustering Configuration
# ============================================================

clustering_config = {
    "selected_algorithm": best_model_entry["algorithm"],
    "selected_parameters": {
        "k": best_model_entry["k"]
    },
    "embedding_model": config.get("model", "unknown"),
    "num_samples": len(metadata),
    "final_metrics": {
        "silhouette": best_model_entry["silhouette"],
        "davies_bouldin": best_model_entry["davies_bouldin"],
        "calinski_harabasz": best_model_entry["calinski_harabasz"]
    }
}

with open(CLUSTERING_DIR / "clustering_config.json", "w", encoding="utf-8") as f:
    json.dump(clustering_config, f, indent=4)

print(" Clustering evaluation completed.")

# ============================================================
# Task 5 - Cluster Analysis
# ============================================================

with open(CLUSTERING_DIR / "cluster_metrics.json", "r", encoding="utf-8") as f:
    best_config = json.load(f)

best_algorithm = best_config["best_model"]["algorithm"]
best_k = best_config["best_model"]["k"]

if best_algorithm == "KMeans":
    # Use the in-memory model matching the winning k (may differ from
    # best_kmeans_k above only if KMeans overall differs - in practice
    # these coincide since both use the same silhouette comparison).
    final_labels = kmeans_models[best_k].labels_

elif best_algorithm == "Hierarchical":
    final_labels = hierarchical_labels

elif best_algorithm == "DBSCAN":
    final_labels = dbscan_labels

else:
    raise ValueError(f"Unknown clustering algorithm: {best_algorithm}")

final_labels = np.asarray(final_labels)

# Save final labels - sample_id + cluster only, as specified.
final_labels_df = pd.DataFrame(
    {
        "sample_id": sample_ids,
        "cluster": final_labels
    }
)

final_labels_df.to_csv(
    CLUSTERING_DIR / "final_cluster_labels.csv",
    index=False
)

# ============================================================
# Representative Samples
# ============================================================
# Distance is computed as cosine distance to the cluster centroid, to stay
# consistent with the cosine metric used throughout evaluation (DBSCAN,
# Hierarchical, and the silhouette score itself).

representatives = []
nearest_examples = []

unique_clusters = sorted(c for c in set(final_labels.tolist()) if c != -1)

for cluster_id in unique_clusters:
    cluster_mask = final_labels == cluster_id
    cluster_embeddings = embeddings[cluster_mask]
    cluster_indices = np.where(cluster_mask)[0]

    centroid = cluster_embeddings.mean(axis=0, keepdims=True)

    distances = cosine_distances(cluster_embeddings, centroid).ravel()

    order = np.argsort(distances)
    nearest_local = order[:5]

    for local_idx in nearest_local:
        sample_idx = int(cluster_indices[local_idx])
        distance = float(distances[local_idx])

        representatives.append(
            {
                "cluster": int(cluster_id),
                "sample_id": sample_idx,
                "distance_to_centroid": distance
            }
        )

        row = metadata.iloc[sample_idx]
        nearest_examples.append(
            {
                "phrase": row.get("phrase", None),
                "level": row.get("level", None),
                "main_category": row.get("main_category", None),
                "sub_category": row.get("sub_category", None),
                "cluster": int(cluster_id),
                "distance": distance
            }
        )

representatives_df = pd.DataFrame(representatives)
representatives_df.to_csv(
    CLUSTERING_DIR / "representative_samples.csv",
    index=False
)

nearest_examples_df = pd.DataFrame(nearest_examples)
nearest_examples_df.to_csv(
    CLUSTERING_DIR / "nearest_examples.csv",
    index=False
)

print(" Representative samples and nearest examples saved.")

# ============================================================
# Final Visualization (reuses existing UMAP projection, not recomputed)
# ============================================================

plt.figure(figsize=(8, 6))
plt.scatter(
    umap_embeddings[:, 0],
    umap_embeddings[:, 1],
    c=final_labels,
    s=10,
    cmap="tab10"
)
plt.title(f"{best_algorithm} Cluster Visualization")
plt.xlabel("UMAP-1")
plt.ylabel("UMAP-2")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "cluster_visualization.png", dpi=300)
plt.close()

print("✅ Task 5 completed.")
print(f"Best model: {best_algorithm} (k={best_k})")
