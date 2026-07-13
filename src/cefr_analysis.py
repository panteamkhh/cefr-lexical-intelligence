# ============================================================
# Compares CEFR proficiency labels against the unsupervised semantic
# clusters discovered in Phase 5, without touching Phase 5's artifacts.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy

# ============================================================
# Project paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = BASE_DIR / "data" / "embeddings"
CLUSTERING_DIR = BASE_DIR / "data" / "clustering"
FIGURE_DIR = BASE_DIR / "data" / "figures"
OUTPUT_DIR = BASE_DIR / "data" / "output"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]

# ============================================================
# Load data
# ============================================================
metadata = pd.read_csv(EMBEDDING_DIR / "metadata.csv")
metadata = metadata.reset_index(drop=True)
metadata["sample_id"] = metadata.index  # 0-based row position, matches
                                         # every clustering artifact below

proj = pd.read_csv(CLUSTERING_DIR / "embedding_projections.csv")

print(f"metadata rows: {len(metadata)}")


def load_clusters(filename):
    labels = pd.read_csv(CLUSTERING_DIR / filename)
    return metadata.merge(labels, on="sample_id", how="left")


def analyze(df, label, save_prefix):
    """Run the full CEFR-vs-cluster analysis for one clustering result."""

    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")

    df = df.copy()
    df["level"] = pd.Categorical(df["level"], categories=CEFR_ORDER, ordered=True)

    # --- Cluster sizes ---
    sizes = df["cluster"].value_counts().sort_index()
    print("Cluster sizes:\n", sizes)

    largest_share = sizes.max() / sizes.sum()
    if largest_share > 0.8:
        print(
            f"  One cluster holds {largest_share:.1%} of all samples. "
            "CEFR-vs-cluster comparison will mostly reflect this single "
            "dominant cluster; treat the smaller clusters as thematic "
            "outlier groups worth inspecting individually, not as a "
            "balanced partition of the vocabulary."
        )

    # --- CEFR distribution per cluster ---
    counts = pd.crosstab(df["cluster"], df["level"])
    pct = pd.crosstab(df["cluster"], df["level"], normalize="index") * 100

    print("\nCEFR counts per cluster:\n", counts)
    print("\nCEFR % per cluster:\n", pct.round(1))

    # --- Purity & entropy per cluster ---
    def purity(row):
        return row.max() / row.sum()

    def cluster_entropy(row):
        p = row / row.sum()
        p = p[p > 0]
        return entropy(p, base=2)

    summary = pd.DataFrame({
        "size": counts.sum(axis=1),
        "dominant_level": counts.idxmax(axis=1),
        "purity": counts.apply(purity, axis=1),
        "entropy_bits": counts.apply(cluster_entropy, axis=1),
    })
    print("\nCluster CEFR summary:\n", summary)

    # --- Mismatch analysis: phrases far from their cluster's CEFR norm ---
    level_to_idx = {lvl: i for i, lvl in enumerate(CEFR_ORDER)}
    level_idx = df["level"].astype(str).map(level_to_idx).astype("Int64")
    dominant_idx = summary["dominant_level"].astype(str).map(level_to_idx).astype("Int64")

    df["level_idx"] = level_idx
    df["cluster_dominant_idx"] = df["cluster"].map(dominant_idx)
    df["level_gap"] = (df["level_idx"] - df["cluster_dominant_idx"]).abs()

    mismatches = df[df["level_gap"] >= 2].sort_values("level_gap", ascending=False)
    print(f"\n{len(mismatches)} phrases flagged (CEFR gap >= 2 levels from cluster norm)")
    print(mismatches[["phrase", "level", "cluster", "level_gap"]].head(15))

    # --- Save outputs ---
    counts.to_csv(OUTPUT_DIR / f"{save_prefix}_crosstab_counts.csv")
    pct.round(2).to_csv(OUTPUT_DIR / f"{save_prefix}_crosstab_pct.csv")
    summary.to_csv(OUTPUT_DIR / f"{save_prefix}_cluster_summary.csv")
    mismatches.to_csv(OUTPUT_DIR / f"{save_prefix}_mismatches.csv", index=False)

    return df, summary


# ============================================================
# Primary analysis: the official Phase 5 winning model
# (whatever algorithm/k was selected by highest silhouette score -
# see data/clustering/clustering_config.json)
# ============================================================
df_final = load_clusters("final_cluster_labels.csv")
df_final, summary_final = analyze(
    df_final, "Primary: final_cluster_labels.csv (Phase 5 winning model)", "cefr_final"
)

# ============================================================
# Secondary analysis: best-of-KMeans (data/clustering/kmeans_labels.csv)
# Kept only as a practical comparison point in case the winning model is
# heavily imbalanced (see the warning printed above, if triggered) -
# this does NOT override or replace the Phase 5 model selection.
# ============================================================
df_kmeans = load_clusters("kmeans_labels.csv")
df_kmeans, summary_kmeans = analyze(
    df_kmeans, "Secondary: kmeans_labels.csv (best-of-KMeans, for comparison)", "cefr_kmeans"
)

# ============================================================
# Data-quality note: phrases with conflicting CEFR levels
# (same phrase, two different levels in the source data - not a
# clustering artifact, carried over from Phase 1)
# ============================================================
level_nunique = metadata.groupby("phrase")["level"].nunique()
conflicting_phrases = level_nunique[level_nunique > 1]
if len(conflicting_phrases) > 0:
    conflict_rows = metadata[metadata["phrase"].isin(conflicting_phrases.index)]
    conflict_rows.sort_values("phrase").to_csv(
        OUTPUT_DIR / "cefr_level_conflicts.csv", index=False
    )
    print(
        f"\n⚠️  {len(conflicting_phrases)} phrases have conflicting CEFR levels "
        f"in the source data ({len(conflict_rows)} rows). "
        f"Saved to {OUTPUT_DIR / 'cefr_level_conflicts.csv'} for manual review - "
        "these are not semantic mismatches, they're a labeling inconsistency."
    )

# ============================================================
# Visualization: UMAP colored by CEFR vs by cluster (primary model)
# ============================================================
plot_df = df_final.merge(proj, on="sample_id", how="left")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for lvl in CEFR_ORDER:
    sub = plot_df[plot_df["level"] == lvl]
    axes[0].scatter(sub["umap_1"], sub["umap_2"], s=4, label=lvl, alpha=0.6)
axes[0].set_title("UMAP colored by CEFR level")
axes[0].legend(markerscale=3, fontsize=8)

for c in sorted(plot_df["cluster"].unique()):
    sub = plot_df[plot_df["cluster"] == c]
    axes[1].scatter(sub["umap_1"], sub["umap_2"], s=4, label=f"cluster {c}", alpha=0.6)
axes[1].set_title("UMAP colored by cluster (final model)")
axes[1].legend(markerscale=3, fontsize=8)

plt.tight_layout()
fig_path = FIGURE_DIR / "cefr_vs_cluster_umap.png"
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved comparison plot to {fig_path}")

print("\n✅ Phase 6 (CEFR Analysis) completed.")
