# ============================================================
# Phase 7 - Semantic Search System
# Turns the Phase 4 embedding space into a query -> top-K retrieval engine,
# with optional filtering by CEFR level and/or category. Reuses
# embeddings.npy / metadata.csv as-is; nothing is recomputed.
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ============================================================
# Project paths
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDING_DIR = BASE_DIR / "data" / "embeddings"

# ============================================================
# Load embedding artifacts (once, at import time)
# ============================================================
_embeddings = np.load(EMBEDDING_DIR / "embeddings.npy")
_metadata = pd.read_csv(EMBEDDING_DIR / "metadata.csv").reset_index(drop=True)

assert _embeddings.shape[0] == len(_metadata), (
    "embeddings.npy and metadata.csv are out of sync - "
    "re-run Phase 4 (embeddings.py) before using search."
)

# Embeddings were saved with normalize_embeddings=True (Phase 4), so plain
# dot product == cosine similarity. Re-normalizing here defensively in case
# a future embeddings.npy isn't pre-normalized.
_norms = np.linalg.norm(_embeddings, axis=1, keepdims=True)
_norms[_norms == 0] = 1.0
_embeddings_normed = _embeddings / _norms

_model = None  # loaded lazily, only when a query actually needs encoding


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _model


def search(query, top_k=10, level=None, category=None, sub_category=None):
    """
    Semantic search over the phrase vocabulary.

    Parameters
    ----------
    query : str
        Free-text query, e.g. "talk about money problems".
    top_k : int
        Number of results to return.
    level : str or list[str], optional
        Restrict results to one or more CEFR levels, e.g. "B2" or ["B1", "B2"].
    category : str or list[str], optional
        Restrict results to one or more main_category values.
    sub_category : str or list[str], optional
        Restrict results to one or more sub_category values.

    Returns
    -------
    pandas.DataFrame with columns: phrase, level, main_category,
    sub_category, similarity - sorted by similarity descending.
    """
    model = _get_model()
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    similarities = _embeddings_normed @ query_vec

    df = _metadata.copy()
    df["similarity"] = similarities

    def _as_list(value):
        if value is None:
            return None
        return [value] if isinstance(value, str) else list(value)

    level_filter = _as_list(level)
    category_filter = _as_list(category)
    sub_category_filter = _as_list(sub_category)

    if level_filter is not None:
        df = df[df["level"].isin(level_filter)]
    if category_filter is not None:
        df = df[df["main_category"].isin(category_filter)]
    if sub_category_filter is not None:
        df = df[df["sub_category"].isin(sub_category_filter)]

    if df.empty:
        return df[["phrase", "level", "main_category", "sub_category", "similarity"]]

    results = df.sort_values("similarity", ascending=False).head(top_k)
    return results[["phrase", "level", "main_category", "sub_category", "similarity"]].reset_index(drop=True)


if __name__ == "__main__":
    demo_queries = [
        ("talk about money problems", {}),
        ("feeling nervous before an exam", {"level": ["B1", "B2"]}),
        ("using a computer or the internet", {"category": "Technology_Digital"}),
    ]

    for q, filters in demo_queries:
        print(f"\nQuery: {q!r}  filters={filters}")
        results = search(q, top_k=5, **filters)
        print(results.to_string(index=False))

    print("\n✅ Phase 7 (Semantic Search System) demo completed.")
