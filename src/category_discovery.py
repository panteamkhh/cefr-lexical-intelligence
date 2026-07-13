# ============================================================
# Phase 8 - Category Discovery
# Interprets the Phase 5 semantic clusters, proposes a signature/name for
# each one from its nearest-to-centroid phrases, and compares the
# discovered grouping against the dataset's original main/sub category
# taxonomy. Nothing from Phase 5 is recomputed.
# ============================================================

from pathlib import Path

import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# ============================================================
# Project paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = BASE_DIR / "data" / "embeddings"
CLUSTERING_DIR = BASE_DIR / "data" / "clustering"
OUTPUT_DIR = BASE_DIR / "data" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Load data
# ============================================================
metadata = pd.read_csv(EMBEDDING_DIR / "metadata.csv").reset_index(drop=True)
metadata["sample_id"] = metadata.index

clusters = pd.read_csv(CLUSTERING_DIR / "final_cluster_labels.csv")
representatives = pd.read_csv(CLUSTERING_DIR / "representative_samples.csv")

df = metadata.merge(clusters, on="sample_id", how="left")
assert df["cluster"].isna().sum() == 0, "Some rows failed to get a cluster label!"

print(f"Total samples: {len(df)} | clusters: {sorted(df['cluster'].unique())}")

# ============================================================
# Cluster sizes + imbalance warning (same check as Phase 6)
# ============================================================
sizes = df["cluster"].value_counts().sort_index()
print("\nCluster sizes:\n", sizes)

largest_share = sizes.max() / sizes.sum()
if largest_share > 0.8:
    print(
        f"⚠️  One cluster holds {largest_share:.1%} of all samples. Category "
        "discovery below will mostly describe that single dominant cluster; "
        "the small clusters are better read as tight thematic outlier groups "
        "than as a full alternative taxonomy."
    )

# ============================================================
# Crosstab: cluster vs main_category
# ============================================================
counts = pd.crosstab(df["cluster"], df["main_category"])
pct = pd.crosstab(df["cluster"], df["main_category"], normalize="index") * 100

print("\nMain-category counts per cluster:\n", counts)
print("\nMain-category %% per cluster:\n", pct.round(1))

counts.to_csv(OUTPUT_DIR / "category_discovery_crosstab_counts.csv")
pct.round(2).to_csv(OUTPUT_DIR / "category_discovery_crosstab_pct.csv")

# ============================================================
# Purity & entropy per cluster (w.r.t. original main_category taxonomy)
# ============================================================
def purity(row):
    return row.max() / row.sum()


def row_entropy(row):
    p = row / row.sum()
    p = p[p > 0]
    return entropy(p, base=2)


cluster_summary = pd.DataFrame({
    "size": counts.sum(axis=1),
    "dominant_main_category": counts.idxmax(axis=1),
    "purity_vs_taxonomy": counts.apply(purity, axis=1),
    "entropy_bits": counts.apply(row_entropy, axis=1),
})

# ============================================================
# Discovered label per cluster: top representative phrases + dominant
# sub_category, used as a human-readable "signature" for the cluster.
# ============================================================
rep_with_text = representatives.merge(
    metadata[["sample_id", "phrase", "main_category", "sub_category"]],
    on="sample_id",
    how="left",
)

signatures = (
    rep_with_text.sort_values(["cluster", "distance_to_centroid"])
    .groupby("cluster")["phrase"]
    .apply(lambda s: ", ".join(s.head(5)))
    .rename("signature_phrases")
)

dominant_sub_category = (
    df.groupby("cluster")["sub_category"]
    .agg(lambda s: s.value_counts().idxmax())
    .rename("dominant_sub_category")
)

cluster_summary = cluster_summary.join(signatures).join(dominant_sub_category)
cluster_summary = cluster_summary[
    ["size", "dominant_main_category", "dominant_sub_category",
     "purity_vs_taxonomy", "entropy_bits", "signature_phrases"]
]

print("\nCluster interpretation:\n")
for cluster_id, row in cluster_summary.iterrows():
    print(f"Cluster {cluster_id} (n={row['size']}):")
    print(f"  Closest existing category : {row['dominant_main_category']} / {row['dominant_sub_category']}")
    print(f"  Purity vs that category   : {row['purity_vs_taxonomy']:.2f}")
    print(f"  Signature phrases         : {row['signature_phrases']}")
    print()

cluster_summary.to_csv(OUTPUT_DIR / "category_discovery_report.csv")

# ============================================================
# Overall alignment metrics: cluster labels vs original taxonomy
# ============================================================
ari = adjusted_rand_score(df["main_category"], df["cluster"])
nmi = normalized_mutual_info_score(df["main_category"], df["cluster"])

print(f"Adjusted Rand Index (cluster vs main_category): {ari:.4f}")
print(f"Normalized Mutual Info (cluster vs main_category): {nmi:.4f}")
print(
    "(Both range 0-1 for NMI, -1 to 1 for ARI; near 0 means the clusters "
    "don't line up with the hand-made taxonomy - i.e. the semantic space "
    "groups phrases differently than the original categories.)"
)

with open(OUTPUT_DIR / "category_discovery_alignment_metrics.txt", "w", encoding="utf-8") as f:
    f.write(f"Adjusted Rand Index (cluster vs main_category): {ari:.4f}\n")
    f.write(f"Normalized Mutual Info (cluster vs main_category): {nmi:.4f}\n")

print("\n✅ Phase 8 (Category Discovery) completed.")
